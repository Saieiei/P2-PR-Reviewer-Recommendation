# -*- coding: utf-8 -*-
#!/usr/bin/env python
"""
BERT + Chroma diff‑retrieval – **v2 (suggestion‑aware)**
-------------------------------------------------------
* Embeds diff + AST + suggestion_text
* Skips trivial “Thanks / LGTM …” results unless they have a suggestion
* Shows suggestion first (fallback to comment)
"""

import argparse
import hashlib
import os
import platform
import re
import shutil
import sys
from pathlib import Path

import ijson
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

# ---------------- CLI ----------------
parser = argparse.ArgumentParser(description="Fast BERT + Chroma diff retrieval")
parser.add_argument("--batch", type=int, default=64)
parser.add_argument("--rebuild", action="store_true")
parser.add_argument("--top-k", type=int, default=3)
parser.add_argument("--disable-ast", action="store_true")
args = parser.parse_args()

DATA_FILE  = "preprocessed_data.json"
PERSIST_DIR = os.path.abspath("./chromadb")

MODEL_NAME = (
    r"C:\sai\HPE\projects\project 2\cloned repo\P2-PR-Reviewer-Recommendation\bert-base-uncased"
    if platform.system() == "Windows"
    else "bert-base-uncased"
)
CLANG_LIB = (
    r"C:\Program Files\LLVM\bin\libclang.dll"
    if platform.system() == "Windows"
    else "/ptmp2/nshashwa/llvm-project/build/lib/libclang.so"
)

# ---------------- AST (optional) ----------------
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
    if not CLANG_AVAILABLE:
        return ""
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".cpp", delete=False) as tmp:
        tmp.write(code.encode())
        path = tmp.name
    try:
        tu = cidx.Index.create().parse(path, args=["-std=c++17"])
        nodes = []
        def visit(n):
            if n.kind.is_declaration():
                nodes.append(f"({n.kind}:{n.spelling})")
            for ch in n.get_children():
                visit(ch)
        visit(tu.cursor)
        return " ".join(nodes)
    finally:
        os.remove(path)

# ---------------- Embedder ----------------
class BERTEmbedder:
    def __init__(self, model_path):
        self.tok = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device).eval()
        print("[DEBUG] BERT on", self.device)

    def _run(self, inputs):
        with torch.no_grad():
            if self.device.type == "cuda":
                with torch.autocast("cuda"):
                    out = self.model(**inputs)
            else:
                out = self.model(**inputs)
        return out.last_hidden_state[:, 0, :]

    def embed_text(self, txt: str) -> np.ndarray:
        inp = self.tok(txt, return_tensors="pt", truncation=True, max_length=512)
        inp = {k: v.to(self.device) for k, v in inp.items()}
        return self._run(inp).cpu().numpy().squeeze(0)

    def embed_batch(self, texts):
        inp = self.tok(texts, return_tensors="pt", truncation=True,
                       padding=True, max_length=512)
        inp = {k: v.to(self.device) for k, v in inp.items()}
        return self._run(inp).cpu().numpy()

embedder = BERTEmbedder(MODEL_NAME)

# ---------------- Chroma ----------------
from chromadb import PersistentClient
client = PersistentClient(path=PERSIST_DIR)
collection = client.get_or_create_collection("bert_embeddings")

# ---------------- Helpers ----------------
def md5(s: str) -> str:
    return hashlib.md5(s.encode()).hexdigest()

_TRIVIAL_RE = re.compile(
    r"^(thanks|thank you|lgtm|looks good to me|done|nit|\+1)[\s!.]*$",
    re.I,
)

def is_trivial(txt: str) -> bool:
    return bool(_TRIVIAL_RE.match((txt or "").strip()))

def stream_json(path):
    with open(path, encoding="utf-8") as fp:
        yield from ijson.items(fp, "item")

# ---------------- Build index ----------------
def _flush(ids, docs, metas, seen):
    if not ids:
        return
    embs = embedder.embed_batch(docs)
    collection.add(ids=ids,
                   embeddings=[e.tolist() for e in embs],
                   documents=docs,
                   metadatas=metas)
    ids.clear(); docs.clear(); metas.clear(); seen.clear()

def build_index():
    global client, collection
    if args.rebuild:
        print("Rebuilding index – wiping", PERSIST_DIR)
        try: client.reset()
        except Exception: pass
        shutil.rmtree(PERSIST_DIR, ignore_errors=True)
        Path(PERSIST_DIR).mkdir(parents=True, exist_ok=True)
        client = PersistentClient(path=PERSIST_DIR)
        collection = client.get_or_create_collection("bert_embeddings")

    if collection.count() and not args.rebuild:
        print(f"Loaded existing index with {collection.count()} vectors")
        return

    ids, docs, metas, seen = [], [], [], set()
    for rec in tqdm(stream_json(DATA_FILE), desc="Indexing", unit="rec"):
        diff   = rec.get("diff_text", "")
        sugg   = rec.get("suggestion_text", "")
        fn     = rec.get("filename", "")
        ast    = parse_cpp_ast(diff) if (CLANG_AVAILABLE and fn.endswith((".c",".cpp",".cc",".h",".hpp"))) else ""
        text   = f"{diff}\n{ast}\n{sugg}"
        cid    = md5(text)
        if cid in seen or collection.get(ids=[cid]).get("ids"):
            continue
        ids.append(cid)
        docs.append(text)
        metas.append({
            "filename": fn,
            "commenter": rec.get("commenter", ""),
            "comment_text": rec.get("comment_text", ""),
            "suggestion_text": sugg,
        })
        seen.add(cid)
        if len(ids) >= args.batch:
            _flush(ids, docs, metas, seen)
    _flush(ids, docs, metas, seen)
    print("Index built – total vectors:", collection.count())

# ---------------- Retrieval ----------------
def retrieve(diff: str, k: int):
    ast = parse_cpp_ast(diff) if CLANG_AVAILABLE else ""
    qvec = embedder.embed_text(f"{diff}\n{ast}").tolist()
    res  = collection.query(query_embeddings=[qvec],
                            n_results=k,
                            include=["distances","metadatas"])
    sims = [1.0 - d for d in res["distances"][0]]
    return list(zip(sims, res["metadatas"][0]))

# ---------------- Interactive ----------------
def read_multiline(prompt="Paste diff/code (end with EOF):"):
    print(prompt)
    lines=[]
    while True:
        try: ln = input()
        except EOFError: return ""
        if ln.strip()=="EOF": break
        lines.append(ln)
    return "\n".join(lines)

def best_payload(meta):
    return meta.get("suggestion_text") or meta.get("comment_text","")

def main():
    build_index()
    while True:
        diff = read_multiline()
        if not diff.strip(): break
        rank = 0
        for score, meta in retrieve(diff, args.top_k*2):  # ask for a few more to filter
            payload = best_payload(meta)
            if not payload.strip(): continue
            if is_trivial(payload) and not meta.get("suggestion_text"):
                continue
            rank += 1
            print(f"#{rank}  sim={score:+.3f}  {meta['filename']}  {meta['commenter']}\n    {payload.splitlines()[0][:120]}")
            if rank >= args.top_k: break
        if input("Test another? (y/n): ").lower()!="y": break
    print("Done.")

if __name__ == "__main__":
    import re  # used by _TRIVIAL_RE
    main()
