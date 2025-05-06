#!/usr/bin/env python3
# retrieval_demo_with_classifier.py

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import sys
import tempfile
from ctypes.util import find_library
from pathlib import Path
from typing import List, Dict

import ijson
import numpy as np
import torch
from chromadb import PersistentClient
from tqdm import tqdm
from transformers import (
    AutoModel,
    AutoTokenizer,
    RobertaForSequenceClassification,
    RobertaTokenizer,
)

# -----------------------------------------------------------------------------  
# ——— CLI & paths ————————————————————————————————————————————————————————  
# -----------------------------------------------------------------------------  
parser = argparse.ArgumentParser(
    description="CodeBERT+Chroma retrieval + CodeBERT classifier filter"
)
parser.add_argument("--batch", type=int, default=256, help="Chroma embedding batch size")
parser.add_argument("--rebuild", action="store_true", help="Wipe & rebuild the Chroma index")
parser.add_argument("--update", action="store_true", help="Incrementally index only new records")
parser.add_argument("--top-k", type=int, default=5, help="How many candidates to retrieve")
parser.add_argument("--disable-ast", action="store_true", help="Skip AST parsing")
parser.add_argument("--rerank", action="store_true", help="Apply cross-encoder reranking")
parser.add_argument(
    "--classifier_dir",
    type=str,
    default="./codebert-finetuned/checkpoint-best",
    help="Path to your fine-tuned CodeBERT classifier (best checkpoint)",
)
parser.add_argument(
    "--threshold", type=float, default=0.51, help="Minimum P(useful) to show a candidate"
)
parser.add_argument(
    "--device",
    type=str,
    default="cuda" if torch.cuda.is_available() else "cpu",
    help="PyTorch device for both retriever & classifier",
)
args = parser.parse_args()

# Base paths
PROJECT_ROOT = os.path.dirname(__file__)
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "codebert-base")
DATA_FILE = "preprocessed_data.json"
PERSIST_DIR = os.path.abspath("./chromadb_uupdate_CodeBERT_db")
AST_CACHE_FILE = "ast_cache.json"
CROSS_ENCODER = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# -----------------------------------------------------------------------------  
# ——— Load & optionally build the Chroma index —————————————————————————  
# -----------------------------------------------------------------------------  
print(f"[INFO] Loading Chroma index at {PERSIST_DIR}")
client = PersistentClient(path=PERSIST_DIR)
collection = client.get_or_create_collection("codebert_embeddings", metadata={"hnsw:space": "cosine"})

# AST caching
try:
    with open(AST_CACHE_FILE, "r", encoding="utf-8") as f:
        _ast_cache: Dict[str, str] = json.load(f)
except:
    _ast_cache = {}

# Attempt to load libclang for ASTs
CLANG_LIB = (
    r"C:\Program Files\LLVM\bin\libclang.dll"
    if platform.system() == "Windows"
    else "/ptmp2/nshashwa/llvm-project/build/lib/libclang.so"
)
CLANG_AVAILABLE = False
if not args.disable_ast:
    try:
        import clang.cindex as cidx
        cidx.Config.set_library_file(CLANG_LIB)
        CLANG_AVAILABLE = True
        print(f"[INFO] AST parser enabled (using libclang at {CLANG_LIB})")
    except Exception as e:
        print(f"[WARN] AST parser unavailable (tried {CLANG_LIB}): {e}")


def parse_cpp_ast(code: str) -> str:
    """Return a declaration-only AST string (cached) for C/C++ snippets."""
    key = hashlib.md5(code.encode()).hexdigest()
    if key in _ast_cache:
        return _ast_cache[key]
    if not CLANG_AVAILABLE or not code.strip():
        _ast_cache[key] = ""
        return ""
    import clang.cindex as cidx
    with tempfile.NamedTemporaryFile(suffix=".cpp", delete=False) as tmp:
        tmp.write(code.encode())
        path = tmp.name
    try:
        tu = cidx.Index.create().parse(path, args=["-std=c++17"])
        changed = set(re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\b", code))
        keep = set()
        def mark(n):
            if n.spelling in changed:
                p = n
                while p:
                    keep.add(p.hash)
                    p = p.semantic_parent
            for c in n.get_children():
                mark(c)
        mark(tu.cursor)
        nodes: List[str] = []
        def visit(n):
            if n.hash in keep and n.kind.is_declaration():
                nodes.append(f"({n.kind}:{n.spelling})")
            for c in n.get_children():
                visit(c)
        visit(tu.cursor)
        ast_str = " ".join(nodes)
        _ast_cache[key] = ast_str
        if len(_ast_cache) % 1000 == 0:
            with open(AST_CACHE_FILE, "w", encoding="utf-8") as f:
                json.dump(_ast_cache, f)
        return ast_str
    finally:
        os.remove(path)


def is_trivial(txt: str, sugg: str) -> bool:
    """Skip one-liner or emoji comments (unless they contain a suggestion)."""
    if sugg:
        return False
    s = (txt or "").strip()
    if len(s.split()) < 5:
        return True
    if not re.search(r"[A-Za-z0-9]", s):
        return True
    return False


def adaptive_threshold(sims: List[float], top_stats=10, alpha=0.7, min_thresh=0.75) -> float:
    top = sorted(sims, reverse=True)[:top_stats]
    mu, sigma = np.mean(top), np.std(top)
    return max(min_thresh, mu + alpha * sigma)


# -----------------------------------------------------------------------------  
# ——— Retriever encoder (CodeBERT) ———————————————————————————————————————  
# -----------------------------------------------------------------------------  
class RetrieverEncoder:
    def __init__(self, model_dir: str, device: str):
        self.tok = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, use_fast=True)
        self.model = AutoModel.from_pretrained(model_dir, local_files_only=True).half()
        self.device = torch.device(device)
        self.model.to(self.device).eval()
        print(f"[INFO] Retriever encoder on {self.device} (FP16)")

    @torch.inference_mode()
    def _run(self, inputs):
        if self.device.type == "cuda":
            with torch.autocast("cuda"):
                out = self.model(**inputs)
        else:
            out = self.model(**inputs)
        return out.last_hidden_state[:, 0, :]

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        inp = self.tok(texts, return_tensors="pt", truncation=True, padding=True, max_length=512).to(self.device)
        vecs = self._run(inp).cpu().numpy()
        return vecs / np.linalg.norm(vecs, axis=-1, keepdims=True)


retriever = RetrieverEncoder(MODEL_PATH, args.device)
if args.rerank:
    from sentence_transformers import CrossEncoder
    reranker = CrossEncoder(CROSS_ENCODER, device=args.device).half()
    print(f"[INFO] Reranker loaded: {CROSS_ENCODER}")
else:
    reranker = None


def build_index():
    global collection
    existing = set(collection.get(include=["ids"])["ids"]) if args.update else set()
    seen = set(existing)

    if args.rebuild:
        print("[INFO] Rebuilding index …")
        shutil.rmtree(PERSIST_DIR, ignore_errors=True)
        os.makedirs(PERSIST_DIR, exist_ok=True)
        client = PersistentClient(path=PERSIST_DIR)
        collection = client.get_or_create_collection("codebert_embeddings", metadata={"hnsw:space": "cosine"})
        existing.clear()
        seen.clear()

    if args.update and existing:
        print(f"[INFO] Incremental mode: skipping {len(existing)} already-indexed items")
    else:
        if collection.count():
            return  # already built

    batch_ids, docs, asts, metas = [], [], [], []
    for rec in tqdm(ijson.items(open(DATA_FILE, encoding="utf-8"), "item"), desc="Indexing"):
        fn = rec.get("filename","").strip()
        diff = (rec.get("diff_text") or "").strip()
        sugg = (rec.get("suggestion_text") or "").strip()
        comm = (rec.get("comment_text") or "").strip()
        ast = parse_cpp_ast(diff) if (CLANG_AVAILABLE and fn.endswith((".cpp",".c",".hpp",".h"))) else ""
        text = (
            f"{fn} ||| {rec.get('title','')} ||| {rec.get('description','')} ||| "
            f"{rec.get('labels','')} ||| {diff}\n"
            f"[AST]{ast}\n[SUGGESTION]{sugg}\n[COMMENT]{comm}"
        )
        key = hashlib.md5(text.encode()).hexdigest()
        if key in seen:
            continue
        seen.add(key)
        batch_ids.append(key); docs.append(text); asts.append(ast)
        metas.append({"filename":fn,"labels":rec.get("labels",""),"suggestion_text":sugg,"comment_text":comm})

        if len(batch_ids) >= args.batch:
            txt_emb = retriever.embed_batch(docs)
            ast_emb = retriever.embed_batch(asts)
            embs = np.concatenate([txt_emb, ast_emb], axis=1).tolist()
            collection.add(ids=batch_ids, embeddings=embs, documents=docs, metadatas=metas)
            batch_ids, docs, asts, metas = [], [], [], []

    if batch_ids:
        txt_emb = retriever.embed_batch(docs)
        ast_emb = retriever.embed_batch(asts)
        embs = np.concatenate([txt_emb, ast_emb], axis=1).tolist()
        collection.add(ids=batch_ids, embeddings=embs, documents=docs, metadatas=metas)

    print(f"[INFO] Index built: {collection.count()} vectors")


# -----------------------------------------------------------------------------  
# ——— Full retrieve + filter pipeline —————————————————————————————————————  
# -----------------------------------------------------------------------------  
def retrieve(diff: str, k: int, fname_filter: str, lbl_filter: str):
    # 1) embed query + optional AST
    ast = parse_cpp_ast(diff) if CLANG_AVAILABLE else ""
    qtxt = f"{diff}\n{ast}"
    txt_emb = retriever.embed_batch([qtxt])[0]
    ast_emb = retriever.embed_batch([ast])[0] if ast else np.zeros_like(txt_emb)
    qvec = np.concatenate([txt_emb, ast_emb])
    qvec /= np.linalg.norm(qvec)

    # 2) query Chroma with optional metadata filters
    where = {}
    if fname_filter:
        where["filename"] = {"$eq": fname_filter}
    if lbl_filter:
        where["labels"] = {"$eq": lbl_filter}
    if where:
        res = collection.query(query_embeddings=[qvec.tolist()], n_results=k*4, where=where, include=["distances","metadatas"])
    else:
        res = collection.query(query_embeddings=[qvec.tolist()], n_results=k*4, include=["distances","metadatas"])
    dists, metas = res["distances"][0], res["metadatas"][0]

    # 3) filter trivial, build list
    items: List[tuple] = []
    for sim, m in zip(dists, metas):
        payload = m.get("suggestion_text") or m.get("comment_text")
        if is_trivial(m.get("comment_text",""), m.get("suggestion_text","")):
            continue
        items.append((sim, m, payload))

    # 4) adaptive threshold
    sims = [sim for sim,_,_ in items]
    th = adaptive_threshold(sims) if sims else 0.0

    # 5) keep only above threshold, pad to k
    cand = [it for it in items if it[0] >= th]
    if len(cand) < k:
        for it in items:
            if len(cand) >= k: break
            if it not in cand:
                cand.append(it)
    cand = cand[:k]

    # 6) optional rerank
    if args.rerank and reranker:
        reranked = []
        for sim, m, payload in cand:
            score = float(reranker.predict([(diff, payload)])[0])
            reranked.append((score, sim, m, payload))
        reranked.sort(reverse=True, key=lambda x: x[0])
        cand = [(sim, m, payload) for _, sim, m, payload in reranked]

    return cand, th


# -----------------------------------------------------------------------------  
# ——— Load your fine-tuned CodeBERT classifier —————————————————————————  
# -----------------------------------------------------------------------------  
print(f"[INFO] Loading classifier from {args.classifier_dir} …")
clf_tokenizer = RobertaTokenizer.from_pretrained(args.classifier_dir, local_files_only=True)
clf_model = RobertaForSequenceClassification.from_pretrained(args.classifier_dir, local_files_only=True)
clf_model.to(args.device).eval()
softmax = torch.nn.Softmax(dim=-1)


# -----------------------------------------------------------------------------  
# ——— Main interactive loop ——————————————————————————————————————————————  
# -----------------------------------------------------------------------------  
def main():
    build_index()
    print(f"\n[INFO] Chroma index ready — retrieving top-{args.top_k} with P(useful) ≥ {args.threshold:.2f}\n")

    while True:
        fname = input("Enter file name filter (or blank): ").strip()
        lbls = input("Enter labels filter (or blank): ").strip()
        print("\nPaste your diff snippet (end with EOF):")
        lines = []
        while True:
            try:
                l = input()
            except EOFError:
                break
            if l.strip().upper() == "EOF":
                break
            lines.append(l)
        query = "\n".join(lines).strip()
        if not query:
            print("[INFO] No input, exiting.")
            break

        candidates, auto_th = retrieve(query, args.top_k, fname, lbls)
        print(f"\n[INFO] Using adaptive retrieval threshold = {auto_th:.3f}\n")

        kept = []
        for sim, m, payload in candidates:
            inputs = clf_tokenizer(payload, return_tensors="pt", truncation=True, padding=True, max_length=200).to(args.device)
            with torch.no_grad():
                logits = clf_model(**inputs).logits
            p1 = float(softmax(logits)[0, 1])
            if p1 >= args.threshold:
                kept.append((sim, m, payload, p1))

        if not kept:
            print("[INFO] No candidates passed the classifier threshold.\n")
        else:
            for i, (sim, m, payload, p1) in enumerate(kept, 1):
                print(f"#{i}  P(useful)={p1:.3f}  sim={sim:.3f}  file={m['filename']} labels={m['labels']}")
                for ln in payload.splitlines():
                    print("    " + ln)
                print()

        if input("Another? (y/n): ").lower() != "y":
            break

    print("Done.")


if __name__ == "__main__":
    main()
