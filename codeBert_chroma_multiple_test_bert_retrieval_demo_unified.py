# -*- coding: utf-8 -*-
#!/usr/bin/env python
"""
BERT/CodeBERT + Chroma Retrieval (v3 – long‑term, high‑quality)
==============================================================
* Default encoder   : microsoft/codebert‑base (code‑aware).
* Distance metric   : cosine (HNSW space=cosine) + L2 fallback.
* Vector normalisation for stable similarities.
* Optional AST      : clang.cindex if libclang available.
* Optional rerank   : cross‑encoder/ms‑marco‑MiniLM‑L‑6‑v2 (‑‑rerank).
* Works on Linux & Windows (WSL) – paths switch automatically.

Run (index build):
    python bert_codebert_chroma_retrieval.py --rebuild --batch 64

Interactive demo:
    python bert_codebert_chroma_retrieval.py --top-k 5 [--rerank]

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
parser = argparse.ArgumentParser("Fast CodeBERT + Chroma diff retrieval")
parser.add_argument("--batch", type=int, default=64, help="embedding batch size")
parser.add_argument("--rebuild", action="store_true", help="wipe & rebuild index")
parser.add_argument("--top-k", type=int, default=5, help="results to show")
parser.add_argument("--disable-ast", action="store_true", help="skip AST parsing even if libclang present")
parser.add_argument("--rerank", action="store_true", help="apply cross‑encoder reranking (slow, but accurate)")
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
    """Return a flat string representation of the C/C++ AST (best‑effort)."""
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
        # CLS token embedding (CodeBERT/Roberta‑style)
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

# Optional cross‑encoder reranker ------------------------------------------------
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
_TRIVIAL_RE = re.compile(
    r"^(thanks|thank you|lgtm|looks good to me|done|nit|\+1)[\s!.]*$", re.I
)


def is_trivial(txt: str) -> bool:
    return bool(_TRIVIAL_RE.match((txt or "").strip()))


def md5(s: str) -> str:
    return hashlib.md5(s.encode()).hexdigest()


# Stream JSON rows lazily -----------------------------------------------------

def stream_json(path):
    with open(path, encoding="utf-8") as fp:
        yield from ijson.items(fp, "item")


# Flush helper for bulk adds ---------------------------------------------------

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


# Build / rebuild index -------------------------------------------------------

def build_index():
    global client, collection
    if args.rebuild:
        print("Rebuilding index – wiping", PERSIST_DIR)
        try:
            client.reset()
        except Exception:
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
        fn = rec.get("filename", "")
        ast = (
            parse_cpp_ast(diff)
            if (CLANG_AVAILABLE and fn.endswith((".c", ".cpp", ".cc", ".h", ".hpp")))
            else ""
        )
        text = f"{fn} ||| {diff}\n{ast}\n{sugg}"
        cid = md5(text)
        if cid in seen or collection.get(ids=[cid]).get("ids"):
            continue
        ids.append(cid)
        docs.append(text)
        metas.append(
            {
                "filename": fn,
                "commenter": rec.get("commenter", ""),
                "labels": rec.get("labels", ""),
                "suggestion_text": sugg,
                "comment_text": rec.get("comment_text", ""),
            }
        )
        seen.add(cid)
        if len(ids) >= args.batch:
            _flush(ids, docs, metas, seen)
    _flush(ids, docs, metas, seen)
    print("Index built – total vectors:", collection.count())


# Retrieve --------------------------------------------------------------------

def retrieve(query_diff: str, k: int) -> List[Tuple[float, dict]]:
    ast = parse_cpp_ast(query_diff) if CLANG_AVAILABLE else ""
    qtxt = f"{query_diff}\n{ast}\n"
    qvec = encoder.embed_text(qtxt).tolist()
    res = collection.query(
        query_embeddings=[qvec], n_results=k, include=["distances", "metadatas"]
    )
    sims = [1.0 - d for d in res["distances"][0]]  # convert cosine distance to similarity
    pairs = list(zip(sims, res["metadatas"][0]))

    # Optional trivial‑comment filter
    pairs = [p for p in pairs if p[1].get("suggestion_text") or not is_trivial(p[1].get("comment_text", ""))]

    # Optional reranking ------------------------------------------------------
    if reranker:
        texts_b = [best_payload(m) for _, m in pairs]
        ce_scores = reranker.predict([(query_diff, t) for t in texts_b])
        pairs = sorted(zip(ce_scores, [m for _, m in pairs]), key=lambda x: -x[0])
        pairs = [(float(score), meta) for score, meta in pairs]
    # Return as many candidates as requested (or more if reranking was applied externally)
    return pairs


# Best payload helper ---------------------------------------------------------

def best_payload(meta: dict) -> str:
    return meta.get("suggestion_text") or meta.get("comment_text", "")


# Multiline reader ------------------------------------------------------------

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


# Main loop -------------------------------------------------------------------

def main():
    build_index()
    while True:
        # Ask for optional meta input: file name and labels.
        input_file = input("Enter file name (optional, press Enter to skip): ").strip()
        input_labels = input("Enter labels (optional, press Enter to skip): ").strip()
        diff = read_multiline("Paste diff/code (end with EOF):")
        if not diff.strip():
            break

        # If metadata (file name/labels) is provided, retrieve more candidates to re‑rank.
        if input_file or input_labels:
            # Increase candidate pool to allow proper re‑ranking.
            candidates = retrieve(diff, args.top_k * 4)
            adjusted_candidates = []
            for score, meta in candidates:
                bonus = 0.0
                # Prioritize file name match strongly.
                if input_file:
                    if input_file.lower() in meta.get("filename", "").lower():
                        bonus += 0.1
                # Add a smaller bonus if label match is found.
                if input_labels:
                    if input_labels.lower() in meta.get("labels", "").lower():
                        bonus += 0.05
                adjusted_candidates.append((score + bonus, meta))
            # Sort candidates based on adjusted score (highest first)
            adjusted_candidates.sort(key=lambda x: -x[0])
            final_results = adjusted_candidates[:args.top_k]
        else:
            # No metadata provided; use default retrieval.
            final_results = retrieve(diff, args.top_k)[:args.top_k]

        # Display the ranked results.
        for rank, (score, meta) in enumerate(final_results, 1):
            payload = best_payload(meta)
            preview = payload.splitlines()[0][:120] if payload else "(no text)"
            print(
                f"#{rank:2d}  sim={score:.3f}  file={meta.get('filename','')}  commenter={meta.get('commenter','')}  labels={meta.get('labels','')}\n    {preview}"
            )
        if input("Test another? (y/n): ").lower() != "y":
            break
    print("Done.")


if __name__ == "__main__":
    main()
