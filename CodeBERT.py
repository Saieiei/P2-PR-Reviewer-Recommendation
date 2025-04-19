# -*- coding: utf-8 -*-
#!/usr/bin/env python
"""
CodeBERT + Chroma Diff Retrieval
=================================
This script uses CodeBERT to embed diffs (code differences) and stores them in a dedicated ChromaDB index
(i.e. a separate “CodeBERT db”). Only the diff text is embedded, and associated comments are stored as metadata.
At query time, the script embeds the provided diff query, performs a cosine similarity search,
and prints the top results (showing similarity score, file path, commenter, and labels).

Usage:
    python CodeBERT.py --rebuild --batch 64 --top-k 5
"""

import argparse
import hashlib
import os
import platform
import re
import shutil
from pathlib import Path
from typing import List, Tuple

import ijson
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

# ----------------------------------------------------------------------------
# CLI arguments
# ----------------------------------------------------------------------------
parser = argparse.ArgumentParser("CodeBERT + Chroma Diff Retrieval")
parser.add_argument("--batch", type=int, default=64, help="embedding batch size")
parser.add_argument("--rebuild", action="store_true", help="wipe & rebuild index")
parser.add_argument("--top-k", type=int, default=5, help="number of results to show")
parser.add_argument("--rerank", action="store_true", help="apply cross-encoder reranking (slow, but accurate)")
args = parser.parse_args()

DATA_FILE = "preprocessed_data.json"

# ----------------------------------------------------------------------------
# Updated Persistent Directory for CodeBERT db
# ----------------------------------------------------------------------------
# Instead of the default "./chromadb", we now use a separate directory for CodeBERT embeddings.
PERSIST_DIR = os.path.abspath("./chromadb_CodeBERT_db")

# ----------------------------------------------------------------------------
# Model configuration
# ----------------------------------------------------------------------------
MODEL_NAME = (
    r"C:\sai\HPE\models\codebert-base" if platform.system() == "Windows" else "microsoft/codebert-base"
)
CROSS_ENCODER_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# ----------------------------------------------------------------------------
# Encoder using CodeBERT (embedding only the diff)
# ----------------------------------------------------------------------------
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
        # Use the CLS token embedding
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

# Optional cross-encoder reranking
if args.rerank:
    from sentence_transformers import CrossEncoder
    reranker = CrossEncoder(CROSS_ENCODER_NAME, device=encoder.device)
    print(f"[DEBUG] Reranker loaded: {CROSS_ENCODER_NAME}")
else:
    reranker = None

# ----------------------------------------------------------------------------
# ChromaDB setup using the separate CodeBERT db
# ----------------------------------------------------------------------------
from chromadb import PersistentClient

client = PersistentClient(path=PERSIST_DIR)
# The collection name is also updated to indicate that these embeddings belong to CodeBERT.
collection = client.get_or_create_collection(
    "codebert_db_embeddings", metadata={"hnsw:space": "cosine"}
)

# ----------------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------------
_TRIVIAL_RE = re.compile(
    r"^(thanks|thank you|lgtm|looks good to me|done|nit|\+1)[\s!.]*$", re.I
)

def is_trivial(txt: str) -> bool:
    return bool(_TRIVIAL_RE.match((txt or "").strip()))

def md5(s: str) -> str:
    return hashlib.md5(s.encode()).hexdigest()

def stream_json(path):
    with open(path, encoding="utf-8") as fp:
        yield from ijson.items(fp, "item")

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

# ----------------------------------------------------------------------------
# Build / rebuild index (embed only the diff text)
# ----------------------------------------------------------------------------
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
            "codebert_db_embeddings", metadata={"hnsw:space": "cosine"}
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
        fn = rec.get("filename", "")
        text = diff  # Only the diff text is embedded
        cid = md5(text + fn)
        if cid in seen or collection.get(ids=[cid]).get("ids"):
            continue
        ids.append(cid)
        docs.append(text)
        metas.append({
            "filename": fn,
            "commenter": rec.get("commenter", ""),
            "labels": rec.get("labels", ""),
            "suggestion_text": rec.get("suggestion_text", ""),
            "comment_text": rec.get("comment_text", ""),
        })
        seen.add(cid)
        if len(ids) >= args.batch:
            _flush(ids, docs, metas, seen)
    _flush(ids, docs, metas, seen)
    print("Index built – total vectors:", collection.count())

# ----------------------------------------------------------------------------
# Retrieve: embed the query diff and perform cosine similarity search
# ----------------------------------------------------------------------------
def retrieve(query_diff: str, k: int) -> List[Tuple[float, dict]]:
    qtxt = query_diff  # Only use the provided diff query text
    qvec = encoder.embed_text(qtxt).tolist()
    res = collection.query(
        query_embeddings=[qvec], n_results=k, include=["distances", "metadatas"]
    )
    sims = [1.0 - d for d in res["distances"][0]]  # cosine similarity: (1 - cosine distance)
    pairs = list(zip(sims, res["metadatas"][0]))

    # Filter out trivial comments, if any (using suggestion_text or comment_text)
    pairs = [p for p in pairs if p[1].get("suggestion_text") or not is_trivial(p[1].get("comment_text", ""))]

    if reranker:
        texts_b = [meta.get("suggestion_text") or meta.get("comment_text", "") for _, meta in pairs]
        ce_scores = reranker.predict([(query_diff, t) for t in texts_b])
        pairs = sorted(zip(ce_scores, [meta for _, meta in pairs]), key=lambda x: -x[0])
        pairs = [(float(score), meta) for score, meta in pairs]
    return pairs

# ----------------------------------------------------------------------------
# Interactive main loop
# ----------------------------------------------------------------------------
def read_multiline(prompt="Paste diff (end with EOF):") -> str:
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

def main():
    build_index()
    while True:
        # Optional filtering: if you want to bias results by file name or labels.
        input_file = input("Enter file name for filtering (optional, press Enter to skip): ").strip()
        input_labels = input("Enter labels for filtering (optional, press Enter to skip): ").strip()
        query_diff = read_multiline("Paste diff (end with EOF):")
        if not query_diff.strip():
            break

        if input_file or input_labels:
            # Retrieve extra candidates if filtering is desired.
            candidates = retrieve(query_diff, args.top_k * 4)
            adjusted_candidates = []
            for score, meta in candidates:
                bonus = 0.0
                if input_file and input_file.lower() in meta.get("filename", "").lower():
                    bonus += 0.1
                if input_labels and input_labels.lower() in meta.get("labels", "").lower():
                    bonus += 0.05
                adjusted_candidates.append((score + bonus, meta))
            adjusted_candidates.sort(key=lambda x: -x[0])
            final_results = adjusted_candidates[:args.top_k]
        else:
            final_results = retrieve(query_diff, args.top_k)[:args.top_k]

        # Display the ranked results.
        for rank, (score, meta) in enumerate(final_results, 1):
            payload = meta.get("suggestion_text") or meta.get("comment_text", "")
            preview = payload.splitlines()[0][:120] if payload else "(no text)"
            print(f"#{rank:2d}  sim={score:.3f}  file={meta.get('filename','')}  commenter={meta.get('commenter','')}  labels={meta.get('labels','')}\n    {preview}")
        if input("Test another? (y/n): ").lower() != "y":
            break
    print("Done.")

if __name__ == "__main__":
    main()

