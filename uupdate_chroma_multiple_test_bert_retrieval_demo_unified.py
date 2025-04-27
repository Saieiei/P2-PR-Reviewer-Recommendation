#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Precision-focused CodeBERT + Chroma diff retrieval (v8, auto-detect libclang)

Features:
- Fast Rust tokenizer (use_fast=True)
- FP16 inference
- AST results cached on disk (if libclang is available)
- Incremental indexing (--update) & full rebuild (--rebuild)
- Batched embedding (default batch=256)
- File-level metadata filtering in Python
- Adaptive similarity threshold
- Drops pure-punctuation payloads (e.g. just "}")
- Prints full payload (suggestion or comment)
"""

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
from ctypes.util import find_library
from pathlib import Path
from typing import List, Dict

import ijson
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from chromadb import PersistentClient

# -----------------------------------------------------------------------------
# Paths & CLI
# -----------------------------------------------------------------------------
PROJECT_ROOT  = os.path.dirname(__file__)
MODEL_PATH    = os.path.join(PROJECT_ROOT, "models", "codebert-base")
DATA_FILE     = "preprocessed_data.json"
PERSIST_DIR   = os.path.abspath("./chromadb_uupdate_CodeBERT_db")
AST_CACHE_FILE= "ast_cache.json"
CROSS_ENCODER = "cross-encoder/ms-marco-MiniLM-L-6-v2"

parser = argparse.ArgumentParser("Precision-focused CodeBERT + Chroma diff retrieval")
parser.add_argument("--batch",      type=int, default=256, help="embedding batch size")
parser.add_argument("--rebuild",    action="store_true", help="wipe & rebuild index")
parser.add_argument("--update",     action="store_true", help="incrementally index new records only")
parser.add_argument("--top-k",      type=int, default=5, help="results to show")
parser.add_argument("--disable-ast",action="store_true", help="skip AST parsing")
parser.add_argument("--rerank",     action="store_true", help="apply cross-encoder reranking")
args = parser.parse_args()

# -----------------------------------------------------------------------------
# Load AST cache
# -----------------------------------------------------------------------------
try:
    with open(AST_CACHE_FILE, "r", encoding="utf-8") as f:
        _ast_cache: Dict[str, str] = json.load(f)
except:
    _ast_cache = {}

# -----------------------------------------------------------------------------
# AST support
# -----------------------------------------------------------------------------
CLANG_LIB = (
    r"C:\Program Files\LLVM\bin\libclang.dll"
    if platform.system() == "Windows"
    else "/ptmp2/nshashwa/llvm-project/build/lib/libclang.so"
)
print(f"Trying to load libclang from: {CLANG_LIB}")
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
        tmp.write(code.encode())
        path = tmp.name
    try:
        import clang.cindex as cidx
        tu = cidx.Index.create().parse(path, args=["-std=c++17"])
        changed = set(re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\b", code))
        keep = set()
        def mark(n):
            if n.spelling in changed:
                p = n
                while p:
                    keep.add(p.hash)
                    p = p.semantic_parent
            for c in n.get_children(): mark(c)
        mark(tu.cursor)
        nodes: List[str] = []
        def visit(n):
            if n.hash in keep and n.kind.is_declaration():
                nodes.append(f"({n.kind}:{n.spelling})")
            for c in n.get_children(): visit(c)
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
# Encoder loading local model only
# -----------------------------------------------------------------------------
class Encoder:
    def __init__(self, model_dir: str):
        self.tok    = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, use_fast=True)
        self.model  = AutoModel   .from_pretrained(model_dir, local_files_only=True).half()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device).eval()
        print(f"[DEBUG] Encoder loaded from {model_dir} on {self.device} (FP16)")

    @torch.inference_mode()
    def _run(self, inputs):
        if self.device.type == "cuda":
            with torch.autocast("cuda"):
                out = self.model(**inputs)
        else:
            out = self.model(**inputs)
        return out.last_hidden_state[:,0,:]

    def _norm(self, vec: np.ndarray) -> np.ndarray:
        return vec / np.linalg.norm(vec, axis=-1, keepdims=True)

    def embed_text(self, txt: str) -> np.ndarray:
        inp = self.tok(txt, return_tensors="pt", truncation=True, padding=True, max_length=512)
        inp = {k: v.to(self.device) for k, v in inp.items()}
        vec = self._run(inp).cpu().numpy()
        return self._norm(vec)[0]

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        inp  = self.tok(texts, return_tensors="pt", truncation=True, padding=True, max_length=512)
        inp  = {k: v.to(self.device) for k, v in inp.items()}
        vecs = self._run(inp).cpu().numpy()
        return self._norm(vecs)

encoder = Encoder(MODEL_PATH)
if args.rerank:
    from sentence_transformers import CrossEncoder
    reranker = CrossEncoder(CROSS_ENCODER, device=encoder.device).half()
    print(f"[DEBUG] Reranker loaded: {CROSS_ENCODER} (FP16)")
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
def is_trivial(txt: str, sugg: str) -> bool:
    if sugg:
        return False
    s = (txt or "").strip()
    if len(s.split()) < 5: return True
    if not re.search(r"[A-Za-z0-9]", s): return True
    return False

def adaptive_threshold(sims: List[float], top_stats=10, alpha=0.7, min_thresh=0.75) -> float:
    top = sorted(sims, reverse=True)[:top_stats]
    mu, sigma = np.mean(top), np.std(top)
    return max(min_thresh, mu + alpha * sigma)

# -----------------------------------------------------------------------------
# Build / Rebuild index
# -----------------------------------------------------------------------------
def build_index():
    global client, collection
    existing, seen = set(), set()

    if args.update and not args.rebuild:
        try:
            existing = set(collection.get(include=["ids"])["ids"])
        except:
            existing = set()
    elif args.rebuild:
        print("Rebuilding index - wiping", PERSIST_DIR)
        shutil.rmtree(PERSIST_DIR, ignore_errors=True)
        Path(PERSIST_DIR).mkdir(parents=True, exist_ok=True)
        client     = PersistentClient(path=PERSIST_DIR)
        collection = client.get_or_create_collection("codebert_embeddings", metadata={"hnsw:space":"cosine"})
    else:
        if collection.count(): return

    ids, docs, asts, metas = [], [], [], []
    for rec in tqdm(ijson.items(open(DATA_FILE, encoding="utf-8"), "item"), desc="Indexing"):
        fn    = rec.get("filename","")
        diff  = rec.get("diff_text","")
        sugg  = rec.get("suggestion_text","")
        comm  = rec.get("comment_text","")
        ast   = parse_cpp_ast(diff) if (CLANG_AVAILABLE and fn.endswith((".cpp",".c",".hpp",".h"))) else ""
        text  = (
            f"{fn} ||| {rec.get('title','')} ||| {rec.get('description','')} ||| "
            f"{rec.get('labels','')} ||| {diff}\n"
            f"[AST]{ast}\n[SUGGESTION]{sugg}\n[COMMENT]{comm}"
        )
        key = hashlib.md5(text.encode()).hexdigest()
        if key in existing or key in seen:
            continue
        seen.add(key)
        ids.append(key); docs.append(text); asts.append(ast)
        metas.append({
            "filename":        fn,
            "commenter":       rec.get("commenter",""),
            "labels":          rec.get("labels",""),
            "suggestion_text": sugg,
            "comment_text":    comm,
        })
        if len(ids) >= args.batch:
            embs = np.concatenate([encoder.embed_batch(docs), encoder.embed_batch(asts)], axis=1)
            collection.add(ids=ids, embeddings=embs.tolist(), documents=docs, metadatas=metas)
            existing.update(ids)
            ids.clear(); docs.clear(); asts.clear(); metas.clear()

    if ids:
        embs = np.concatenate([encoder.embed_batch(docs), encoder.embed_batch(asts)], axis=1)
        collection.add(ids=ids, embeddings=embs.tolist(), documents=docs, metadatas=metas)

    print("Index complete - total vectors:", collection.count())
    with open(AST_CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(_ast_cache, f)

# -----------------------------------------------------------------------------
# Retrieval w/ file-level filtering
# -----------------------------------------------------------------------------
def retrieve(query_diff: str, k: int, file_filter: str):
    # Embed query + optional AST
    ast_used = False
    if CLANG_AVAILABLE:
        ast = parse_cpp_ast(query_diff); ast_used = bool(ast)
    else:
        ast = ""
    qtxt    = f"{query_diff}\n{ast}\n"
    txt_vec = encoder.embed_text(qtxt)
    ast_vec = encoder.embed_text(ast) if ast_used else np.zeros_like(txt_vec)
    qvec    = np.concatenate([txt_vec, ast_vec])
    qvec   /= np.linalg.norm(qvec, keepdims=True)

    # Query Chroma
    if file_filter:
        res = collection.query(
            query_embeddings=[qvec.tolist()],
            n_results=k*4,
            where={"filename":{"$eq":file_filter}},
            include=["distances","metadatas","embeddings"]
        )
        dists, metas, embds = res["distances"][0], res["metadatas"][0], res["embeddings"][0]
        if not dists:
            gr = collection.query(query_embeddings=[qvec.tolist()], n_results=k*4,
                                   include=["distances","metadatas","embeddings"])
            dists, metas, embds = gr["distances"][0], gr["metadatas"][0], gr["embeddings"][0]
    else:
        gr = collection.query(query_embeddings=[qvec.tolist()], n_results=k*4,
                              include=["distances","metadatas","embeddings"])
        dists, metas, embds = gr["distances"][0], gr["metadatas"][0], gr["embeddings"][0]

    # Score/filter
    items = []
    for sim, m, demb in zip(dists, metas, embds):
        payload = m.get("suggestion_text","") or m.get("comment_text","")
        if is_trivial(m.get("comment_text",""), m.get("suggestion_text","")):
            continue
        arr = np.array(demb)
        dt, at = arr[:txt_vec.shape[0]], arr[txt_vec.shape[0]:]
        items.append({
            "sim": sim,
            "txt_sim": float(np.dot(txt_vec, dt)),
            "ast_sim": float(np.dot(ast_vec, at)) if ast_used else 0.0,
            "meta": m,
            "payload": payload
        })

    th = adaptive_threshold([it["sim"] for it in items])
    filtered = [it for it in items if it["sim"] >= th]
    if len(filtered) < k:
        for it in items:
            if len(filtered) >= k: break
            if it not in filtered:
                filtered.append(it)
    final = filtered[:k]

    # Rerank optional
    if args.rerank and reranker:
        scores = reranker.predict([(query_diff, it["payload"]) for it in final])
        for it, s in zip(final, scores):
            it["rerank_score"] = float(s)
        final.sort(key=lambda x: x.get("rerank_score",0), reverse=True)

    return final, th, ast_used

# -----------------------------------------------------------------------------
# Main loop
# -----------------------------------------------------------------------------
def main():
    build_index()
    while True:
        fname = input("Enter file name (optional): ").strip()
        lbls  = input("Enter labels (optional): ").strip()
        print("Paste diff (end with EOF):")
        lines = []
        while True:
            try:
                ln = input()
            except EOFError:
                break
            if ln.strip() == "EOF":
                break
            lines.append(ln)
        diff = "\n".join(lines)
        if not diff.strip(): break

        results, threshold, ast_used = retrieve(diff, args.top_k, fname)
        print(f"Using adaptive threshold: {threshold:.3f}\n")

        for i, it in enumerate(results, 1):
            base_sim = 1.0 - it["sim"]
            fb = 0.1 if (fname and fname.lower() in it["meta"]["filename"].lower()) else 0.0
            lb = 0.05 if (lbls and lbls.lower() in it["meta"]["labels"].lower()) else 0.0
            score = base_sim + fb + lb
            factors = [f"sim={base_sim:.3f}", f"text={it['txt_sim']:.3f}", f"ast={it['ast_sim']:.3f}"]
            if fb: factors.append(f"file_boost={fb:.2f}")
            if lb: factors.append(f"label_boost={lb:.2f}")
            if args.rerank: factors.append(f"rerank={it.get('rerank_score',0):.3f}")

            m = it["meta"]
            print(f"#{i} score={score:.3f} ({', '.join(factors)}) "
                  f"file={m['filename']} commenter={m['commenter']} labels={m['labels']}")
            for line in it["payload"].splitlines():
                print("    "+line)
            print()

        if input("Another? (y/n): ").lower() != "y":
            break
    print("Done.")

if __name__ == "__main__":
    main()
