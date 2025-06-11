#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
retrieval_gcbert_feed.py – GraphCodeBERT + classifier + Chroma retrieval **with user feedback loop**

This script is a **superset** of the original `retrieval_gcbert.py` (commit 2025‑06‑11).
It preserves *all* existing functionality and library calls and **does not touch the
stored embeddings**.  Two lightweight extensions add an interactive feedback loop:

1.  After the usual ranking, the script prompts the user to mark which of the shown
    suggestions/comments were actually helpful (e.g. `1,4,5`).  The mapping from
    canonical‑MD5‑of‑text → *fav_count* is stored in `fav_scores.json` next to the
    script.
2.  During ranking the tiny **tie‑breaker weight** `FAV_WEIGHT = 0.001` converts
    `fav_count` into `fav_score = fav_count × FAV_WEIGHT`, and a
        `final_score = similarity + fav_score`
    is computed.  Candidates are displayed in **descending `final_score`** order,
    together with *fav_count* and *final_score*.

That’s it – the vector index is never rebuilt or updated unless the user supplies
`--rebuild/--update`, exactly like the base script.
"""

from __future__ import annotations
import argparse, hashlib, json, math, os, shutil, sys, tempfile
from pathlib import Path
from typing import List, Dict, Any, Set

import torch, ijson, tqdm  # third‑party deps identical to original script
from chromadb import PersistentClient
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification

# ───────────────────────────────────────────────────────────────────────
# 0)  Feedback constants
# ───────────────────────────────────────────────────────────────────────
FAV_FILE   = "fav_scores.json"   # persistent store ⩘ hash(text) → integer count
FAV_WEIGHT = 1e-3                # tiny tie‑breaker weight requested by user

# Helpers to (de)serialise persistent fav‑scores ------------------------

def _load_fav() -> Dict[str, int]:
    if os.path.exists(FAV_FILE):
        try:
            with open(FAV_FILE, "r", encoding="utf-8") as fp:
                return {k: int(v) for k, v in json.load(fp).items()}
        except Exception:
            pass  # corrupt → start fresh
    return {}


def _save_fav(d: Dict[str, int]):
    tmp = FAV_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fp:
        json.dump(d, fp, ensure_ascii=False, indent=2)
    os.replace(tmp, FAV_FILE)

_fav_scores: Dict[str, int] = _load_fav()

# ───────────────────────────────────────────────────────────────────────
# 1)  Filters & helpers from process_data_pr.py  (unchanged)
# ───────────────────────────────────────────────────────────────────────
try:
    import process_data_pr as pdp
except ModuleNotFoundError:
    sys.path.append(Path(__file__).resolve().parent.as_posix())
    import process_data_pr as pdp  # type: ignore

is_trivial              = pdp.is_trivial
is_generic              = pdp.is_generic
is_suggestion           = pdp.is_suggestion
has_technical_reference = pdp.has_technical_reference
is_design               = pdp.is_design
LENGTH_THRESHOLD        = pdp.LENGTH_THRESHOLD

# ───────────────────────────────────────────────────────────────────────
# 2)  CLI
# ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser("GC‑BERT retrieval (with feedback loop)")
parser.add_argument("--batch",          type=int, default=64)
parser.add_argument("--rebuild",        action="store_true")
parser.add_argument("--update",         action="store_true")
parser.add_argument("--top-k",          type=int, default=15,               # user asked for 15
                   help="number of suggestions to *display* (ranking uses extra) ")
parser.add_argument("--threshold",      type=float, default=0.38)
parser.add_argument("--device",         choices=["cuda", "cpu"], default=None)
parser.add_argument("--data-file",      default="preprocessed_data.json")
parser.add_argument("--encoder_dir",    default="./models/graphcodebert-base/")
parser.add_argument("--classifier_dir", default="./sweep_results/graphcodebert/lr2e-5_bs64_ep5/checkpoint-best")
parser.add_argument("--clang-lib",      default="/ptmp2/moresair/cloned_repo/P2-PR-Reviewer-Recommendation/llvm-project/build/lib/libclang.so")
parser.add_argument("--persist-dir",    default="./chromadb_gcbert")
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
        toks: List[str] = []
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
# 4)  Encoder & classifier (unchanged)
# ───────────────────────────────────────────────────────────────────────

device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))

_tok_enc = AutoTokenizer.from_pretrained(args.encoder_dir)
_enc      = AutoModel.from_pretrained   (args.encoder_dir).to(device).eval()
print(f"[INFO] Encoder on {device}")

@torch.no_grad()
def _embed(x: List[str] | str):
    if isinstance(x, str):
        x = [x]
    batch = _tok_enc(x, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
    with torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
        v = _enc(**batch).last_hidden_state[:, 0, :]
    return v.cpu().numpy()

_clf     = AutoModelForSequenceClassification.from_pretrained(args.classifier_dir).to(device).eval()
_tok_clf = _tok_enc
print("[INFO] Classifier loaded")

@torch.no_grad()
def _p_keep(x: List[str] | str):
    if isinstance(x, str):
        x = [x]
    batch = _tok_clf(x, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
    logits = _clf(**batch).logits
    if logits.shape[-1] == 1:
        return torch.sigmoid(logits).squeeze(-1).cpu().numpy()
    return torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()

# ───────────────────────────────────────────────────────────────────────
# 5)  Chroma helpers (unchanged except cosine metric patch already present)
# ───────────────────────────────────────────────────────────────────────
client: PersistentClient | None = None
col     = None  # late‑bound

def _md5(txt: str) -> str:  # stable key for fav‑score
    return hashlib.md5(txt.encode()).hexdigest()

def _keep_record(comm: str, sugg: str) -> bool:
    if sugg:
        return True
    if not comm:
        return False
    if is_trivial(comm) or is_generic(comm):
        return False
    if len(comm) <= LENGTH_THRESHOLD and not (
        has_technical_reference(comm) or is_suggestion(comm) or is_design(comm)
    ):
        return False
    return True

def _subdir_token(path: str) -> str:
    parts = Path(path).parts[:2]
    return f"SUBDIR:{'/'.join(parts)}" if parts else ""

# Build / update index (identical to original except already‑patched cosine) ----

def _flush(ids, docs, metas):
    if ids:
        col.add(ids=ids,
                embeddings=[v.tolist() for v in _embed(docs)],
                documents=docs,
                metadatas=metas)
        ids.clear(); docs.clear(); metas.clear()


def _stream_json(fp):
    with open(fp, encoding="utf-8") as f:
        yield from ijson.items(f, "item")


def build_or_update():
    global client, col
    if args.rebuild and os.path.exists(args.persist_dir):
        shutil.rmtree(args.persist_dir, ignore_errors=True)
    Path(args.persist_dir).mkdir(parents=True, exist_ok=True)
    client = PersistentClient(path=args.persist_dir)
    col    = client.get_or_create_collection(name="gcbert_embeds", metadata={"hnsw:space": "cosine"})

    if not (args.rebuild or args.update):  # query‑only mode
        print(f"[INFO] Chroma collection (cosine) loaded – {col.count():,} vectors")
        return

    existing: Set[str] = set()
    if args.update and col.count():
        try:
            existing = set(col.get(include=[])["ids"])
            print(f"[INFO] {len(existing):,} existing vectors found")
        except Exception as e:
            print(f"[WARN] couldn’t fetch existing ids: {e}")

    ids: List[str] = []
    docs: List[str] = []
    metas: List[Dict[str, Any]] = []
    seen: Set[str] = set()
    action = "rebuild" if args.rebuild else "update"
    print(f"[INFO] {action.capitalize()} index (cosine) from {args.data_file} …")

    for rec in tqdm.tqdm(_stream_json(args.data_file), unit="rec"):
        diff = rec.get("diff_text", "")
        sugg = rec.get("suggestion_text", "")
        comm = rec.get("comment_text", "")
        if not _keep_record(comm, sugg):
            continue
        fn     = rec.get("filename", "")
        asttxt = parse_cpp_ast(diff) if (CLANG_AVAILABLE and fn.endswith((".c", ".cpp", ".h", ".hpp", ".cc", ".cxx"))) else ""
        body   = sugg if sugg else comm
        doc    = f"{_subdir_token(fn)}\n{diff}\n{asttxt}\n{body}".strip()
        hid    = _md5(doc)
        if hid in seen or (args.update and hid in existing):
            continue

        ids.append(hid)
        docs.append(doc)
        metas.append({"filename": fn, "comment_text": comm, "suggestion_text": sugg})
        seen.add(hid)
        if len(ids) >= args.batch:
            _flush(ids, docs, metas)
    _flush(ids, docs, metas)
    print(f"[INFO] {action.capitalize()} complete – index size {col.count():,}")

# ───────────────────────────────────────────────────────────────────────
# 6)  Retrieval helpers (unchanged apart from cosine explanation)
# ───────────────────────────────────────────────────────────────────────

def _retrieve(diff: str, k: int, fname: str | None = None):
    sub  = _subdir_token(fname) if fname else ""
    ast  = parse_cpp_ast(diff) if CLANG_AVAILABLE else ""
    qtxt = f"{sub}\n{diff}\n{ast}".strip()
    qvec = _embed(qtxt)[0].tolist()
    res  = col.query(query_embeddings=[qvec], n_results=k, include=["distances", "metadatas"])
    return list(zip(res["distances"][0], res["metadatas"][0]))  # [(dist, meta), …]


def _best_text(meta):
    return meta.get("suggestion_text") or meta.get("comment_text", "")

# Canonical text for duplicate suppression & fav‑hash -------------------

def _canonical(txt: str) -> str:
    return " ".join(txt.split()).lower()

# ───────────────────────────────────────────────────────────────────────
# 7)  Simple I/O helpers (unchanged)
# ───────────────────────────────────────────────────────────────────────

def _read_filename() -> str | None:
    fn = input("File path (optional, press <enter> to skip): ").strip()
    return fn or None

def _read_diff() -> str:
    print("Paste diff (end with line 'EOF'):")
    buf: List[str] = []
    while True:
        try:
            ln = input()
        except EOFError:
            break
        if ln.strip().upper() == "EOF":
            break
        buf.append(ln)
    return "\n".join(buf)

# ───────────────────────────────────────────────────────────────────────
# 8)  Interactive main loop  **WITH FEEDBACK**
# ───────────────────────────────────────────────────────────────────────

def main():
    build_or_update()
    print(f"\n[INFO] Ready – top-{args.top_k}, P(keep) ≥ {args.threshold}\n")

    while True:
        fname = _read_filename()
        diff  = _read_diff()
        if not diff.strip():
            print("Empty diff – bye!")
            break

        try:
            raw = _retrieve(diff, args.top_k * 10, fname)  # grab extra → survives dedup
        except RuntimeError as e:
            print(e)
            break

        # Pass 1: filter duplicates & low‑probability, compute scores ----------
        seen_texts: Set[str] = set()
        cand: List[Dict[str, Any]] = []

        for dist_sq, meta in raw:
            txt = _best_text(meta)
            if not txt:
                continue
            if _canonical(txt) in seen_texts:
                continue  # duplicate suppression
            if not _keep_record(meta.get("comment_text", ""), meta.get("suggestion_text", "")):
                continue
            pk = _p_keep(txt)[0]
            if pk < args.threshold:
                continue

            seen_texts.add(_canonical(txt))

            # Similarity from L2 of cosine distance ---------------------------
            l2  = math.sqrt(max(dist_sq, 0.0))
            sim = 1.0 / (1.0 + l2)  # ∈ [0,1]

            key = _md5(_canonical(txt))
            fav = _fav_scores.get(key, 0)
            final = sim + fav * FAV_WEIGHT

            cand.append({
                "text": txt,
                "meta": meta,
                "sim": sim,
                "pk": pk,
                "fav": fav,
                "final": final,
            })

        if not cand:
            print("¯\\_(ツ)_/¯  – nothing survived the filters")
            if not input("Another diff? (y/N): ").lower().startswith("y"):
                break
            else:
                continue

        # Sort & show ---------------------------------------------------------
        cand.sort(key=lambda d: (-d["final"], -d["sim"]))
        top = cand[: args.top_k]

        for idx, c in enumerate(top, start=1):
            meta = c["meta"]
            ttype = "suggestion" if meta.get("suggestion_text") else "comment"
            print(f"--- Recommendation #{idx} ---")
            print(f"  File        : {meta['filename']}")
            print(f"  Type        : {ttype}")
            print(f"  Similarity  : {c['sim']:.4f}")
            print(f"  P(keep)     : {c['pk']:.4f}")
            print(f"  fav_count   : {c['fav']}")
            print(f"  final_score : {c['final']:.4f}")
            print("  Text:")
            for ln in c["text"].strip().splitlines():
                print(f"    {ln}")
            print()

        # ------------------- optional feedback collection -------------------
        resp = input("Enter numbers of helpful suggestions (comma / space‑sep, empty to skip): ").strip()
        if resp:
            sels = {int(tok) for tok in resp.replace(",", " ").split() if tok.isdigit()}
            for sel in sels:
                if 1 <= sel <= len(top):
                    txt_key = _md5(_canonical(top[sel - 1]["text"]))
                    _fav_scores[txt_key] = _fav_scores.get(txt_key, 0) + 1
                    print(f"👍  feedback recorded for #{sel}")
            _save_fav(_fav_scores)

        if not input("Another diff? (y/N): ").lower().startswith("y"):
            break

    print("GG!")


if __name__ == "__main__":
    main()
