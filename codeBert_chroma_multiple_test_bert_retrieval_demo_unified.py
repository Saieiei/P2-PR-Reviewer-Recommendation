# -*- coding: utf-8 -*-
#!/usr/bin/env python
"""
BERT/CodeBERT + Chroma Retrieval (v3 – precision-focused)

This script builds a Chroma index of (diff + AST + suggestion) embeddings and
retrieves high-precision, context-aware comments for a given diff.

Run (index build):
    python codeBert_chroma_multiple_test_bert_retrieval_demo_unified.py --rebuild --batch 64

Interactive demo:
    python codeBert_chroma_multiple_test_bert_retrieval_demo_unified.py --top-k 5 [--rerank]
"""

from __future__ import annotations
import argparse
import hashlib
import os
import platform
import re
import shutil
import sys
from pathlib import Path
from typing import List, Tuple

import ijson
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

# ---------------------------------------------------------------------------
# ---------------------------- CLI ------------------------------------------
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser("Precision-focused CodeBERT + Chroma diff retrieval")
parser.add_argument("--batch", type=int, default=64, help="embedding batch size")
parser.add_argument("--rebuild", action="store_true", help="wipe & rebuild index")
parser.add_argument("--top-k", type=int, default=5, help="results to show")
parser.add_argument("--disable-ast", action="store_true", help="skip AST parsing even if libclang present")
parser.add_argument("--rerank", action="store_true", help="apply cross-encoder reranking (more accurate)")
args = parser.parse_args()

DATA_FILE = "preprocessed_data.json"
PERSIST_DIR = os.path.abspath("./chromadb")

# ---------------------------------------------------------------------------
# ---------------------------- Models ---------------------------------------
# ---------------------------------------------------------------------------
MODEL_NAME = (
    r"C:\sai\HPE\models\codebert-base" if platform.system() == "Windows" else "microsoft/codebert-base"
)
CROSS_ENCODER_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# ---------------------------------------------------------------------------
# ---------------------------- AST Support ----------------------------------
# ---------------------------------------------------------------------------
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
        print("INFO: AST parser enabled (libclang)")
    except Exception as e:
        print("WARNING: AST parser unavailable:", e)


def parse_cpp_ast(code: str) -> str:
    """Return a flat string representation of the C/C++ AST (best-effort)."""
    if not CLANG_AVAILABLE:
        return ""
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".cpp", delete=False) as tmp:
        tmp.write(code.encode())
        path = tmp.name
    try:
        tu = cidx.Index.create().parse(path, args=["-std=c++17"])
        nodes: List[str] = []
        def visit(node):
            if node.kind.is_declaration():
                nodes.append(f"({node.kind}:{node.spelling})")
            for ch in node.get_children():
                visit(ch)
        visit(tu.cursor)
        return " ".join(nodes)
    finally:
        os.remove(path)

# ---------------------------------------------------------------------------
# ---------------------------- Embedder -------------------------------------
# ---------------------------------------------------------------------------
class Encoder:
    def __init__(self, model_name: str):
        self.tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device).eval()
        print(f"[DEBUG] Encoder loaded: {model_name} on {self.device}")

    @torch.inference_mode()
    def _run(self, inputs):
        if self.device.type == "cuda":
            with torch.autocast("cuda"):
                out = self.model(**inputs)
        else:
            out = self.model(**inputs)
        return out.last_hidden_state[:, 0, :]

    def _norm(self, vec: np.ndarray) -> np.ndarray:
        return vec / np.linalg.norm(vec, ord=2, axis=-1, keepdims=True)

    def embed_text(self, txt: str) -> np.ndarray:
        inp = self.tok(txt, return_tensors="pt", truncation=True, max_length=512)
        inp = {k: v.to(self.device) for k, v in inp.items()}
        vec = self._run(inp).cpu().numpy()
        return self._norm(vec)[0]

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        inp = self.tok(texts, return_tensors="pt", truncation=True, padding=True, max_length=512)
        inp = {k: v.to(self.device) for k, v in inp.items()}
        vecs = self._run(inp).cpu().numpy()
        return self._norm(vecs)

encoder = Encoder(MODEL_NAME)

# Optional cross-encoder reranker
if args.rerank:
    from sentence_transformers import CrossEncoder
    reranker = CrossEncoder(CROSS_ENCODER_NAME, device=encoder.device)
    print(f"[DEBUG] Reranker loaded: {CROSS_ENCODER_NAME}")
else:
    reranker = None

# ---------------------------------------------------------------------------
# ---------------------------- Chroma ---------------------------------------
# ---------------------------------------------------------------------------
from chromadb import PersistentClient
client = PersistentClient(path=PERSIST_DIR)
collection = client.get_or_create_collection(
    "codebert_embeddings", metadata={"hnsw:space": "cosine"}
)

# ---------------------------------------------------------------------------
# ---------------------------- Helpers --------------------------------------
# ---------------------------------------------------------------------------
_TRIVIAL_PATTERNS = [
    r"^(thanks|thank you|lgtm|looks good to me|done|nit|\+1)[\s!.]*$",
    r"^\s*good catch\b.*$",
    r"^\s*ah[, ]*great[.!]*$",
    r"^\s*due to\b.*$",
]
_TRIVIAL_RE = [re.compile(pat, re.I) for pat in _TRIVIAL_PATTERNS]

def is_trivial(txt: str) -> bool:
    s = (txt or "").strip()
    if len(s.split()) < 6:
        return True
    if any(p.match(s) for p in _TRIVIAL_RE):
        return True
    if not re.search(r"[A-Za-z0-9]", s):
        return True
    return False

def is_technical(txt: str) -> bool:
    return bool(re.search(r"[`;(){}\[\]]|[A-Za-z0-9_]+[A-Z][A-Za-z0-9_]*", txt or ""))

def best_payload(meta: dict) -> str:
    return meta.get("suggestion_text") or meta.get("comment_text", "")

# ---------------------------------------------------------------------------
# ---------------------------- Stream JSON ----------------------------------
# ---------------------------------------------------------------------------
def stream_json(path):
    with open(path, encoding="utf-8") as fp:
        yield from ijson.items(fp, "item")

# ---------------------------------------------------------------------------
# ---------------------------- Bulk Add Flush -------------------------------
# ---------------------------------------------------------------------------
def _flush(ids, docs, metas, seen):
    if not ids:
        return
    embs = encoder.embed_batch(docs)
    collection.add(
        ids=ids,
        embeddings=[e.tolist() for e in embs],
        documents=docs,
        metadatas=metas,
    )
    ids.clear(); docs.clear(); metas.clear(); seen.clear()

# ---------------------------------------------------------------------------
# ---------------------------- Build / Rebuild Index ------------------------
# ---------------------------------------------------------------------------
def build_index():
    global client, collection
    if args.rebuild:
        print("Rebuilding index – wiping", PERSIST_DIR)
        try:
            client.reset()
        except:
            pass
        shutil.rmtree(PERSIST_DIR, ignore_errors=True)
        Path(PERSIST_DIR).mkdir(parents=True, exist_ok=True)
        client = PersistentClient(path=PERSIST_DIR)
        collection = client.get_or_create_collection(
            "codebert_embeddings", metadata={"hnsw:space": "cosine"}
        )
    if collection.count() and not args.rebuild:
        print(f"Loaded existing index with {collection.count()} vectors")
        return

    ids: List[str] = []
    docs: List[str] = []
    metas: List[dict] = []
    seen: set[str] = set()
    for rec in tqdm(stream_json(DATA_FILE), desc="Indexing", unit="rec"):
        diff = rec.get("diff_text", "")
        sugg = rec.get("suggestion_text", "")
        fn   = rec.get("filename", "")
        ast  = parse_cpp_ast(diff) if (CLANG_AVAILABLE and fn.endswith((".c", ".cpp", ".cc", ".h", ".hpp"))) else ""
        text = f"{fn} ||| {diff}\n{ast}\n{sugg}"
        cid  = hashlib.md5(text.encode()).hexdigest()
        if cid in seen or collection.get(ids=[cid]).get("ids"):
            continue
        ids.append(cid)
        docs.append(text)
        metas.append({
            "filename": fn,
            "commenter": rec.get("commenter", ""),
            "labels": rec.get("labels", ""),
            "suggestion_text": sugg,
            "comment_text": rec.get("comment_text", ""),
        })
        seen.add(cid)
        if len(ids) >= args.batch:
            _flush(ids, docs, metas, seen)
    _flush(ids, docs, metas, seen)
    print("Index built – total vectors:", collection.count())

# ---------------------------------------------------------------------------
# ---------------------------- Retrieve -------------------------------------
# ---------------------------------------------------------------------------
def retrieve(query_diff: str, k: int) -> List[Tuple[float, dict]]:
    ast = parse_cpp_ast(query_diff) if CLANG_AVAILABLE else ""
    qtxt = f"{query_diff}\n{ast}\n"
    qvec = encoder.embed_text(qtxt).tolist()

    res = collection.query(
        query_embeddings=[qvec],
        n_results=k * 4,
        include=["distances", "metadatas"]
    )
    sims = [1.0 - d for d in res["distances"][0]]
    pairs = list(zip(sims, res["metadatas"][0]))

    # 1) Drop trivial
    pairs = [ (s,m) for s,m in pairs
              if m.get("suggestion_text") or not is_trivial(m.get("comment_text","")) ]
    # 2) Require code/context tokens
    pairs = [ (s,m) for s,m in pairs if is_technical(best_payload(m)) ]
    # 3) Deduplicate
    seen_texts = set(); unique = []
    for s,m in pairs:
        txt = best_payload(m).strip()
        if txt in seen_texts: continue
        seen_texts.add(txt)
        unique.append((s,m))
    # 4) Similarity threshold
    threshold = 0.85
    final = [(s,m) for s,m in unique if s >= threshold]
    # 5) Pad if fewer than k
    if len(final) < k:
        for s,m in unique:
            if len(final) >= k: break
            if (s,m) not in final: final.append((s,m))
    return final[:k]

# ---------------------------------------------------------------------------
# ---------------------------- Multiline Reader -----------------------------
# ---------------------------------------------------------------------------
def read_multiline(prompt="Paste diff/code (end with EOF):") -> str:
    print(prompt)
    lines: List[str] = []
    while True:
        try:
            ln = input()
        except EOFError:
            return ""
        if ln.strip() == "EOF":
            break
        lines.append(ln)
    return "\n".join(lines)

# ---------------------------------------------------------------------------
# ---------------------------- Main Loop ------------------------------------
# ---------------------------------------------------------------------------

def main():
    build_index()
    while True:
        input_file   = input("Enter file name (optional, press Enter to skip): ").strip()
        input_labels = input("Enter labels (optional, press Enter to skip): ").strip()
        diff         = read_multiline()
        if not diff.strip():
            break

        if input_file or input_labels:
            candidates = retrieve(diff, args.top_k * 4)
            adjusted = []
            for s,m in candidates:
                bonus = 0.0
                if input_file and input_file.lower() in m.get("filename",""").lower(): bonus += 0.1
                if input_labels and input_labels.lower() in m.get("labels",""").lower():  bonus += 0.05
                adjusted.append((s+bonus, m))
            adjusted.sort(key=lambda x:-x[0])
            results = adjusted[:args.top_k]
        else:
            results = retrieve(diff, args.top_k)

        for idx, (score, meta) in enumerate(results, 1):
            txt = best_payload(meta)
            preview = txt.splitlines()[0][:120] if txt else "(no text)"
            print(f"#{idx:2d} sim={score:.3f} file={meta.get('filename')} commenter={meta.get('commenter')} labels={meta.get('labels')}\n    {preview}")
        if input("Test another? (y/n): ").lower() != "y":
            break
    print("Done.")

if __name__ == "__main__":
    main()
