#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Precision-focused CodeBERT + Chroma diff retrieval (v6, with AST‐score output and full‐payload printing)

Features:
- Fast Rust tokenizer (use_fast=True)
- FP16 inference
- AST results cached on disk
- Incremental indexing (--update) & full rebuild (--rebuild)
- Batched embedding (default batch=256)
- Adaptive similarity threshold
- Explainability: prints base, text‐embedding, AST‐embedding, file/label boosts, rerank
- Prints the **entire** suggestion/comment payload, not just a truncated preview
"""

from __future__ import annotations
import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import sys
import tempfile
from pathlib import Path
from typing import List, Dict

import ijson
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from chromadb import PersistentClient

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser("Precision-focused CodeBERT + Chroma diff retrieval")
parser.add_argument("--batch",      type=int,      default=256, help="embedding batch size")
parser.add_argument("--rebuild",    action="store_true",   help="wipe & rebuild index")
parser.add_argument("--update",     action="store_true",   help="incrementally index new records only")
parser.add_argument("--top-k",      type=int,      default=5,   help="results to show")
parser.add_argument("--disable-ast",action="store_true",   help="skip AST parsing")
parser.add_argument("--rerank",     action="store_true",   help="apply cross-encoder reranking")
args = parser.parse_args()

DATA_FILE      = "preprocessed_data.json"
PERSIST_DIR    = os.path.abspath("./chromadb")
AST_CACHE_FILE = "ast_cache.json"

# -----------------------------------------------------------------------------
# AST cache load
# -----------------------------------------------------------------------------
try:
    with open(AST_CACHE_FILE, encoding="utf-8") as f:
        _ast_cache: Dict[str,str] = json.load(f)
except Exception:
    _ast_cache = {}

# -----------------------------------------------------------------------------
# Models & AST support
# -----------------------------------------------------------------------------
MODEL_NAME         = "microsoft/codebert-base"
CROSS_ENCODER_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

CLANG_LIB = (
    r"C:\Program Files\LLVM\bin\libclang.dll" if platform.system()=="Windows"
    else "/ptmp2/nshashwa/llvm-project/build/lib/libclang.so"
)
CLANG_AVAILABLE = False
if not args.disable_ast:
    try:
        import clang.cindex as cidx
        cidx.Config.set_library_file(CLANG_LIB)
        CLANG_AVAILABLE = True
        print("INFO: AST parser enabled")
    except Exception as e:
        print("WARNING: AST parser unavailable:", e)

def parse_cpp_ast(code: str) -> str:
    key = hashlib.md5(code.encode()).hexdigest()
    if key in _ast_cache:
        return _ast_cache[key]
    if not CLANG_AVAILABLE or not code.strip():
        _ast_cache[key] = ""
        return ""
    with tempfile.NamedTemporaryFile(suffix=".cpp", delete=False) as tmp:
        tmp.write(code.encode()); path = tmp.name
    try:
        tu = cidx.Index.create().parse(path, args=["-std=c++17"])
        changed = set(re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\b", code))
        keep = set()
        def mark(node):
            if node.spelling in changed:
                p = node
                while p:
                    keep.add(p.hash)
                    p = p.semantic_parent
            for c in node.get_children(): mark(c)
        mark(tu.cursor)
        nodes: List[str] = []
        def visit(node):
            if node.hash in keep and node.kind.is_declaration():
                nodes.append(f"({node.kind}:{node.spelling})")
            for c in node.get_children(): visit(c)
        visit(tu.cursor)
        ast_str = " ".join(nodes)
        _ast_cache[key] = ast_str
        if len(_ast_cache) % 1000 == 0:
            with open(AST_CACHE_FILE, "w", encoding="utf-8") as f:
                json.dump(_ast_cache, f)
        return ast_str
    finally:
        os.remove(path)

# -----------------------------------------------------------------------------
# Encoder
# -----------------------------------------------------------------------------
class Encoder:
    def __init__(self, model_name: str):
        self.tok   = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=True)
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True).half()
        self.device= torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device).eval()
        print(f"[DEBUG] Encoder loaded: {model_name} on {self.device} (FP16)")

    @torch.inference_mode()
    def _run(self, inputs):
        if self.device.type=="cuda":
            with torch.autocast("cuda"):
                out = self.model(**inputs)
        else:
            out = self.model(**inputs)
        return out.last_hidden_state[:,0,:]

    def _norm(self, vec: np.ndarray) -> np.ndarray:
        return vec / np.linalg.norm(vec, axis=-1, keepdims=True)

    def embed_text(self, txt: str) -> np.ndarray:
        inp = self.tok(txt, return_tensors="pt", truncation=True, padding=True, max_length=512)
        inp = {k:v.to(self.device) for k,v in inp.items()}
        vec = self._run(inp).cpu().numpy()
        return self._norm(vec)[0]

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        inp = self.tok(texts, return_tensors="pt", truncation=True, padding=True, max_length=512)
        inp = {k:v.to(self.device) for k,v in inp.items()}
        vecs= self._run(inp).cpu().numpy()
        return self._norm(vecs)

encoder = Encoder(MODEL_NAME)
if args.rerank:
    from sentence_transformers import CrossEncoder
    reranker = CrossEncoder(CROSS_ENCODER_NAME, device=encoder.device)
    reranker.model = reranker.model.half()
    print(f"[DEBUG] Reranker loaded: {CROSS_ENCODER_NAME} (FP16)")
else:
    reranker = None

# -----------------------------------------------------------------------------
# Chroma client
# -----------------------------------------------------------------------------
client     = PersistentClient(path=PERSIST_DIR)
collection = client.get_or_create_collection("codebert_embeddings", metadata={"hnsw:space":"cosine"})

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
_TRIVIAL_PATTERNS = [
    r"^(thanks|thank you|lgtm|looks good to me|done|nit|\+1)[\s!.]*$",
    r"^\s*good catch\b.*$",
    r"^\s*ah[, ]*great[.!]*$",
    r"^\s*due to\b.*$",
]
_TRIVIAL_RE = [re.compile(p, re.I) for p in _TRIVIAL_PATTERNS]

def is_trivial(txt: str, sugg: str) -> bool:
    if sugg:
        return False
    s = (txt or "").strip()
    if len(s.split()) < 8:
        return True
    if any(p.match(s) for p in _TRIVIAL_RE):
        return True
    if not re.search(r"[A-Za-z0-9]", s):
        return True
    return False

def adaptive_threshold(sims: List[float], top_stats=10, alpha=0.7, min_thresh=0.75) -> float:
    top = sorted(sims, reverse=True)[:top_stats]
    mu, sigma = np.mean(top), np.std(top)
    return max(min_thresh, mu + alpha * sigma)

# -----------------------------------------------------------------------------
# Build / Rebuild / Update Index
# -----------------------------------------------------------------------------
def build_index():
    global client, collection
    existing_ids = set()
    seen: set = set()

    if args.update and not args.rebuild:
        try:
            existing_ids = set(collection.get(include=["ids"])["ids"])
        except:
            existing_ids = set()
        print(f"Updating index: skipping {len(existing_ids)} existing vectors")

    elif args.rebuild:
        print("Rebuilding index – wiping", PERSIST_DIR)
        shutil.rmtree(PERSIST_DIR, ignore_errors=True)
        Path(PERSIST_DIR).mkdir(parents=True, exist_ok=True)
        client     = PersistentClient(path=PERSIST_DIR)
        collection = client.get_or_create_collection("codebert_embeddings", metadata={"hnsw:space":"cosine"})

    else:
        if collection.count():
            print(f"Loaded existing index with {collection.count()} vectors")
            return

    ids, docs, asts, metas = [], [], [], []
    for rec in tqdm(ijson.items(open(DATA_FILE, encoding="utf-8"), "item"), desc="Indexing"):
        fn     = rec.get("filename","")
        diff   = rec.get("diff_text","")
        title  = rec.get("title","")
        desc   = rec.get("description","")
        labels = rec.get("labels","")
        sugg   = rec.get("suggestion_text","")
        comm   = rec.get("comment_text","")

        ast = parse_cpp_ast(diff) if (CLANG_AVAILABLE and fn.endswith((".c",".cpp",".h",".hpp"))) else ""

        text = (
            f"{fn} ||| {title} ||| {desc} ||| {labels} ||| {diff}\n"
            f"[AST]{ast}\n[SUGGESTION]{sugg}\n[COMMENT]{comm}"
        )
        key = hashlib.md5(text.encode()).hexdigest()

        if key in existing_ids or key in seen:
            continue
        seen.add(key)

        ids.append(key)
        docs.append(text)
        asts.append(ast)
        metas.append({
            "filename":        fn,
            "commenter":       rec.get("commenter",""),
            "labels":          labels,
            "suggestion_text": sugg,
            "comment_text":    comm,
        })

        if len(ids) >= args.batch:
            embs = np.concatenate([encoder.embed_batch(docs), encoder.embed_batch(asts)], axis=1)
            collection.add(ids=ids, embeddings=embs.tolist(), documents=docs, metadatas=metas)
            existing_ids.update(ids)
            ids.clear(); docs.clear(); asts.clear(); metas.clear()

    if ids:
        embs = np.concatenate([encoder.embed_batch(docs), encoder.embed_batch(asts)], axis=1)
        collection.add(ids=ids, embeddings=embs.tolist(), documents=docs, metadatas=metas)

    print("Index complete – total vectors:", collection.count())
    with open(AST_CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(_ast_cache, f)

# -----------------------------------------------------------------------------
# Retrieval
# -----------------------------------------------------------------------------
def retrieve(query_diff: str, k: int):
    ast = parse_cpp_ast(query_diff) if CLANG_AVAILABLE else ""
    ast_used = bool(ast)

    # embed query text & AST
    qtxt     = f"{query_diff}\n{ast}\n"
    text_vec = encoder.embed_text(qtxt)
    ast_vec  = encoder.embed_text(ast) if ast_used else np.zeros_like(text_vec)

    # concat & normalize → 1536-d
    qvec = np.concatenate([text_vec, ast_vec])
    qvec /= np.linalg.norm(qvec, axis=-1, keepdims=True)

    # query Chroma (include embeddings to split)
    res = collection.query(
        query_embeddings=[qvec.tolist()],
        n_results=k * 4,
        include=["distances","metadatas","embeddings"]
    )
    sims       = [1.0 - d for d in res["distances"][0]]
    metas      = res["metadatas"][0]
    doc_embeds = res["embeddings"][0]

    items = []
    for sim, m, demb in zip(sims, metas, doc_embeds):
        sugg = m.get("suggestion_text","")
        comm = m.get("comment_text","")
        if is_trivial(comm, sugg):
            continue

        # split document embedding
        demb_arr   = np.array(demb)
        dt_emb     = demb_arr[: text_vec.shape[0]]
        ast_emb    = demb_arr[text_vec.shape[0] :]

        txt_sim = float(np.dot(text_vec, dt_emb))
        ast_sim = float(np.dot(ast_vec, ast_emb)) if ast_used else 0.0

        # choose payload
        if sugg:
            payload = sugg
        else:
            payload = comm
            if not re.match(r"^[A-Z].*\.$", payload):
                if not re.search(r"[`;(){}\[\]]|[A-Za-z0-9_]+[A-Z]", payload):
                    continue

        items.append({
            "sim":     sim,
            "txt_sim": txt_sim,
            "ast_sim": ast_sim,
            "meta":    m,
            "payload": payload
        })

    # adaptive threshold & pad
    th       = adaptive_threshold([it["sim"] for it in items])
    filtered = [it for it in items if it["sim"] >= th]
    if len(filtered) < k:
        for it in items:
            if len(filtered) >= k: break
            if it not in filtered: filtered.append(it)
    final = filtered[:k]

    # optional cross‑encoder rerank
    if args.rerank and reranker:
        pairs  = [(query_diff, it["payload"]) for it in final]
        scores = reranker.predict(pairs)
        for it, r in zip(final, scores):
            it["rerank_score"] = float(r)
        final.sort(key=lambda x: x.get("rerank_score",0), reverse=True)

    return final, th, ast_used

# -----------------------------------------------------------------------------
# Main loop
# -----------------------------------------------------------------------------
def main():
    build_index()
    while True:
        file_in = input("Enter file name (optional): ").strip()
        lbls_in = input("Enter labels (optional): ").strip()
        print("Paste diff (end with EOF):")
        lines = []
        while True:
            try:
                ln = input()
            except EOFError:
                break
            if ln.strip()=="EOF":
                break
            lines.append(ln)
        diff = "\n".join(lines)
        if not diff.strip():
            break

        results, threshold, ast_used = retrieve(diff, args.top_k)
        print(f"Using adaptive threshold: {threshold:.3f}\n")

        for idx, it in enumerate(results, 1):
            base    = it["sim"]
            txt_s   = it["txt_sim"]
            ast_s   = it["ast_sim"]
            fb = lb = 0.0
            if file_in and file_in.lower() in it["meta"]["filename"].lower():
                fb = 0.1
            if lbls_in and lbls_in.lower() in it["meta"]["labels"].lower():
                lb = 0.05
            final_score = base + fb + lb

            factors = [
                f"base={base:.3f}",
                f"text={txt_s:.3f}",
                f"ast={ast_s:.3f}"
            ]
            if fb:
                factors.append(f"file_boost={fb:.2f}")
            if lb:
                factors.append(f"label_boost={lb:.2f}")
            if args.rerank:
                factors.append(f"rerank={it.get('rerank_score',0):.3f}")

            m = it["meta"]
            print(
                f"#{idx} score={final_score:.3f} ({', '.join(factors)}) "
                f"file={m['filename']} commenter={m['commenter']} labels={m['labels']}"
            )
            # --- FULL PAYLOAD PRINTING ---
            for line in it["payload"].splitlines():
                print("    " + line)
            print()

        if input("Another? (y/n): ").lower() != "y":
            break
    print("Done.")

if __name__=="__main__":
    main()
