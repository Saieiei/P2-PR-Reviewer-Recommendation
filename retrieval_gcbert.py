#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
retrieval_gcbert.py  –  GraphCodeBERT + classifier + Chroma retrieval

🏷  Changes in this version (2025-05-23)
------------------------------------------------
• Duplicate suppression in the interactive loop:
    – canonicalises comment/suggestion text with
      `" ".join(txt.split()).lower()` and keeps only the first hit.
"""

from __future__ import annotations
import argparse, hashlib, os, shutil, sys, tempfile, math
from pathlib import Path
from typing import List, Dict, Any
import torch, ijson, tqdm

# ───────────────────────────────────────────────────────────────────────
# 1)  Filter helpers from process_data_pr.py
# ───────────────────────────────────────────────────────────────────────
try:
    import process_data_pr as pdp
except ModuleNotFoundError:
    sys.path.append(Path(__file__).resolve().parent.as_posix())
    import process_data_pr as pdp            # type: ignore

is_trivial              = pdp.is_trivial
is_generic              = pdp.is_generic
is_suggestion           = pdp.is_suggestion
has_technical_reference = pdp.has_technical_reference
is_design               = pdp.is_design
LENGTH_THRESHOLD        = pdp.LENGTH_THRESHOLD

# ───────────────────────────────────────────────────────────────────────
# 2)  CLI
# ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser("GC-BERT retrieval demo")
parser.add_argument("--batch", type=int, default=64)
parser.add_argument("--rebuild", action="store_true")
parser.add_argument("--update",  action="store_true")
parser.add_argument("--top-k",   type=int, default=10)
parser.add_argument("--threshold", type=float, default=0.47)
parser.add_argument("--device", choices=["cuda", "cpu"], default=None)
parser.add_argument("--data-file", default="preprocessed_data.json")
parser.add_argument("--encoder_dir",     default="./models/graphcodebert-base/")
parser.add_argument("--classifier_dir",  default="./sweep_results/graphcodebert/lr2e-5_bs64_ep5/checkpoint-best")
parser.add_argument("--clang-lib",       default="/ptmp2/moresair/cloned_repo/P2-PR-Reviewer-Recommendation/llvm-project/build/lib/libclang.so")
parser.add_argument("--persist-dir",     default="./chromadb_gcbert")
args = parser.parse_args()
if args.rebuild and args.update:
    sys.exit("❌  --rebuild and --update cannot be combined")

# ───────────────────────────────────────────────────────────────────────
# 3)  Optional libclang for small AST token boost (unchanged)
# ───────────────────────────────────────────────────────────────────────
CLANG_AVAILABLE = False
if not os.getenv("DISABLE_AST"):
    try:
        import clang.cindex as cidx
        cidx.Config.set_library_file(args.clang_lib)
        CLANG_AVAILABLE = True
        print("[INFO] AST enabled (libclang loaded)")
    except Exception as e:
        print(f"[WARN] libclang unavailable – AST disabled: {e}")

def parse_cpp_ast(code: str) -> str:
    if not CLANG_AVAILABLE or len(code) < 10:
        return ""
    with tempfile.NamedTemporaryFile(suffix=".cpp", delete=False, mode="w", encoding="utf-8") as tmp:
        tmp.write(code)
        fname = tmp.name
    try:
        tu = cidx.Index.create().parse(fname, args=["-std=c++17"])
        toks: list[str] = []
        def walk(n):
            if n.kind.is_declaration():
                toks.append(f"({n.kind.name}:{n.spelling})")
            for ch in n.get_children():
                walk(ch)
        walk(tu.cursor)
        return " ".join(toks)
    except Exception as e:
        print(f"[WARN] AST parse failed: {e}")
        return ""
    finally:
        os.remove(fname)

# ───────────────────────────────────────────────────────────────────────
# 4)  Encoder & classifier
# ───────────────────────────────────────────────────────────────────────
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))

tok_enc = AutoTokenizer.from_pretrained(args.encoder_dir)
enc     = AutoModel.from_pretrained   (args.encoder_dir).to(device).eval()
print(f"[INFO] Encoder on {device}")

@torch.no_grad()
def embed(x: list[str] | str):
    if isinstance(x, str): x = [x]
    batch = tok_enc(x, padding=True, truncation=True,
                    max_length=512, return_tensors="pt").to(device)
    with torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
        v = enc(**batch).last_hidden_state[:, 0, :]
    return v.cpu().numpy()

clf = AutoModelForSequenceClassification.from_pretrained(
        args.classifier_dir).to(device).eval()
tok_clf = tok_enc
print("[INFO] Classifier loaded")

@torch.no_grad()
def p_keep(x: list[str] | str):
    if isinstance(x, str): x = [x]
    batch = tok_clf(x, padding=True, truncation=True,
                    max_length=512, return_tensors="pt").to(device)
    logits = clf(**batch).logits
    if logits.shape[-1] == 1:
        return torch.sigmoid(logits).squeeze(-1).cpu().numpy()
    return torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()

# ───────────────────────────────────────────────────────────────────────
# 5)  ChromaDB helpers
# ───────────────────────────────────────────────────────────────────────
from chromadb import PersistentClient
client: PersistentClient | None = None
col    = None                   # late-bound

def md5(txt: str) -> str: return hashlib.md5(txt.encode()).hexdigest()

def keep_record(comm: str, sugg: str) -> bool:
    if sugg: return True
    if not comm: return False
    if is_trivial(comm) or is_generic(comm): return False
    if len(comm) <= LENGTH_THRESHOLD and not (
        has_technical_reference(comm) or is_suggestion(comm) or is_design(comm)
       ):
        return False
    return True

def subdir_token(path: str) -> str:
    parts = Path(path).parts[:2]
    return f"SUBDIR:{'/'.join(parts)}" if parts else ""

# ───────────────────────────────────────────────────────────────────────
# 6)  Build / update index  – unchanged
# ───────────────────────────────────────────────────────────────────────
def flush(ids, docs, metas):
    if ids:
        col.add(ids=ids,
                embeddings=[v.tolist() for v in embed(docs)],
                documents=docs,
                metadatas=metas)
        ids.clear(); docs.clear(); metas.clear()

def stream_json(fp):               # memory-light JSON streamer
    with open(fp, encoding="utf-8") as f:
        yield from ijson.items(f, "item")

def build_or_update():
    global client, col
    if args.rebuild and os.path.exists(args.persist_dir):
        shutil.rmtree(args.persist_dir, ignore_errors=True)
    Path(args.persist_dir).mkdir(parents=True, exist_ok=True)
    client = PersistentClient(path=args.persist_dir)
    col    = client.get_or_create_collection("gcbert_embeds")  # default L2

    # quick exit if we’re only querying
    if not (args.rebuild or args.update):
        print(f"[INFO] Chroma collection loaded – {col.count():,} vectors")
        return

    existing = set()
    if args.update and col.count():
        try:
            existing = set(col.get(include=[])["ids"])
            print(f"[INFO] {len(existing):,} existing vectors found")
        except Exception as e:
            print(f"[WARN] couldn’t fetch existing ids: {e}")

    ids, docs, metas, seen = [], [], [], set()
    action = "rebuild" if args.rebuild else "update"
    print(f"[INFO] {action.capitalize()} index from {args.data_file} …")

    for rec in tqdm.tqdm(stream_json(args.data_file), unit="rec"):
        diff = rec.get("diff_text", "")
        sugg = rec.get("suggestion_text", "")
        comm = rec.get("comment_text", "")
        if not keep_record(comm, sugg): continue

        fn     = rec.get("filename", "")
        asttxt = parse_cpp_ast(diff) if (CLANG_AVAILABLE and fn.endswith((".c",".cpp",".h",".hpp",".cc",".cxx"))) else ""
        body   = sugg if sugg else comm
        doc    = f"{subdir_token(fn)}\n{diff}\n{asttxt}\n{body}".strip()
        hid    = md5(doc)
        if hid in seen or (args.update and hid in existing): continue

        ids.append(hid)
        docs.append(doc)
        metas.append({"filename": fn,
                      "comment_text": comm,
                      "suggestion_text": sugg})
        seen.add(hid)
        if len(ids) >= args.batch: flush(ids, docs, metas)
    flush(ids, docs, metas)
    print(f"[INFO] {action.capitalize()} complete – index size {col.count():,}")

# ───────────────────────────────────────────────────────────────────────
# 7)  Retrieval helpers
# ───────────────────────────────────────────────────────────────────────
def retrieve(diff: str, k: int, fname: str | None = None):
    sub  = subdir_token(fname) if fname else ""
    ast  = parse_cpp_ast(diff) if CLANG_AVAILABLE else ""
    qtxt = f"{sub}\n{diff}\n{ast}".strip()
    qvec = embed(qtxt)[0].tolist()
    res  = col.query(query_embeddings=[qvec], n_results=k,
                     include=["distances", "metadatas"])
    return list(zip(res["distances"][0], res["metadatas"][0]))

def best_text(meta): return meta.get("suggestion_text") or meta.get("comment_text","")

# ───────────────────────────────────────────────────────────────────────
# 8)  Interactive loop  (duplicate suppression added here)
# ───────────────────────────────────────────────────────────────────────
def read_filename() -> str | None:
    fn = input("File path (optional, press <enter> to skip): ").strip()
    return fn or None

def read_diff() -> str:
    print("Paste diff (end with line 'EOF'):")
    buf = []
    while True:
        try:
            ln = input()
        except EOFError:
            break
        if ln.strip().upper() == "EOF":
            break
        buf.append(ln)
    return "\n".join(buf)

def canonical(txt: str) -> str:
    """Minimal normalisation for duplicate detection."""
    return " ".join(txt.split()).lower()

def main():
    build_or_update()
    print(f"\n[INFO] Ready – top-{args.top_k}, P(keep) ≥ {args.threshold}\n")

    while True:
        fname = read_filename()
        diff  = read_diff()
        if not diff.strip():
            print("Empty diff – bye!")
            break

        try:
            res = retrieve(diff, args.top_k * 8, fname)   # fetch extra → survives dedup
        except RuntimeError as e:
            print(e); break

        shown      = 0
        seen_texts: set[str] = set()

        for dist_sq, meta in res:
            txt   = best_text(meta)
            if not txt: continue
            if canonical(txt) in seen_texts:  # ← DUPLICATE SKIP
                continue
            if not keep_record(meta.get("comment_text",""), meta.get("suggestion_text","")):
                continue
            pk = p_keep(txt)[0]
            if pk < args.threshold:
                continue

            seen_texts.add(canonical(txt))
            shown += 1

            l2   = math.sqrt(max(dist_sq, 0.0))
            sim  = 1.0 / (1.0 + l2)           # bounded [0,1]

            ttype = "suggestion" if meta.get("suggestion_text") else "comment"
            print(f"--- Recommendation #{shown} ---")
            print(f"  File       : {meta['filename']}")
            print(f"  Type       : {ttype}")
            print(f"  Distance   : {dist_sq:.4f}")
            print(f"  Similarity : {sim:.4f}")
            print(f"  P(keep)    : {pk:.4f}")
            print("  Text:")
            for ln in txt.strip().splitlines():
                print(f"    {ln}")
            print()
            if shown >= args.top_k:
                break

        if shown == 0:
            print("¯\\_(ツ)_/¯  – nothing survived the filters")

        if not input("Another diff? (y/N): ").lower().startswith("y"):
            break
    print("GG!")

if __name__ == "__main__":
    main()

