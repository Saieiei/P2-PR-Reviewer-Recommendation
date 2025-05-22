#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GraphCodeBERT + classifier + Chroma retrieval (with optional SUBDIR token)

• Embeds diff + AST [+ SUBDIR:<top-2 path parts>] [+ suggestion/comment]
• Reuses filtering helpers from process_data_pr.py
• Prints raw distance and cosine similarity (1 - distance)
• Indicates text type (comment vs suggestion) in output
• Supports --rebuild, --update, --top-k, --threshold, --device
"""

from __future__ import annotations
import argparse, hashlib, os, shutil, sys, tempfile
from pathlib import Path
from typing import List, Dict, Any

# import filter suite
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

# CLI args
parser = argparse.ArgumentParser("GC-BERT retrieval demo")
parser.add_argument("--batch", type=int, default=64)
parser.add_argument("--rebuild", action="store_true")
parser.add_argument("--update", action="store_true")
parser.add_argument("--top-k", type=int, default=10)
parser.add_argument("--threshold", type=float, default=0.47)
parser.add_argument("--device", choices=["cuda", "cpu"], default=None)
parser.add_argument("--data-file", default="preprocessed_data.json")
parser.add_argument("--encoder_dir", default="./models/graphcodebert-base/")
parser.add_argument("--classifier_dir", default="./sweep_results/graphcodebert/lr2e-5_bs64_ep5/checkpoint-best")
parser.add_argument("--clang-lib", default="/ptmp2/moresair/cloned_repo/P2-PR-Reviewer-Recommendation/llvm-project/build/lib/libclang.so")
parser.add_argument("--persist-dir", dest="persist_dir", default="./chromadb_gcbert")
args = parser.parse_args()
if args.rebuild and args.update:
    sys.exit("❌ --rebuild and --update cannot be combined")

# AST helper
CLANG_AVAILABLE = False
if not os.getenv("DISABLE_AST"):
    try:
        import clang.cindex as cidx
        cidx.Config.set_library_file(args.clang_lib)
        CLANG_AVAILABLE = True
        print("[INFO] AST enabled")
    except Exception as e:
        print(f"[WARN] libclang absent – AST disabled: {e}")

def parse_cpp_ast(code: str) -> str:
    if not CLANG_AVAILABLE or len(code) < 10:
        return ""
    with tempfile.NamedTemporaryFile(suffix=".cpp", delete=False, mode="w", encoding="utf-8") as tmp:
        tmp.write(code)
        fname = tmp.name
    try:
        tu = cidx.Index.create().parse(fname, args=["-std=c++17"])
        tokens: List[str] = []
        def walk(node):
            if node.kind.is_declaration():
                tokens.append(f"({node.kind.name}:{node.spelling})")
            for ch in node.get_children():
                walk(ch)
        walk(tu.cursor)
        return " ".join(tokens)
    except Exception as e:
        print(f"[WARN] AST parse failed: {e}")
        return ""
    finally:
        os.remove(fname)

# Embedding & classifier setup
import torch, ijson, tqdm # ijson and tqdm are used in build_or_update
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification

device = torch.device(args.device if args.device else (
    "cuda" if torch.cuda.is_available() else "cpu"))

# Encoder
tok_enc = AutoTokenizer.from_pretrained(args.encoder_dir)
enc     = AutoModel.from_pretrained(args.encoder_dir).to(device).eval()
print(f"[INFO] Encoder on {device}")
@torch.no_grad()
def embed(x: List[str] | str):
    if isinstance(x, str): x = [x]
    batch = tok_enc(x, padding=True, truncation=True,
                    max_length=512, return_tensors="pt").to(device)
    with torch.autocast(device_type=device.type, enabled=(device.type=="cuda")):
        v = enc(**batch).last_hidden_state[:, 0, :]
    return v.cpu().numpy()

# Classifier (reuse tokenizer)
tok_clf = tok_enc
clf     = AutoModelForSequenceClassification.from_pretrained(
    args.classifier_dir).to(device).eval()
print("[INFO] Classifier loaded")
@torch.no_grad()
def p_keep(x: List[str] | str):
    if isinstance(x, str): x = [x]
    batch = tok_clf(x, padding=True, truncation=True,
                    max_length=512, return_tensors="pt").to(device)
    logits = clf(**batch).logits
    if logits.shape[-1] == 1:
        return torch.sigmoid(logits).squeeze(-1).cpu().numpy()
    return torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()

# ChromaDB setup
from chromadb import PersistentClient
client: PersistentClient | None = None # Will be initialized in build_or_update
col    = None # Will be initialized in build_or_update

def md5(txt: str) -> str:
    return hashlib.md5(txt.encode('utf-8')).hexdigest()

# Record filter
def keep_record(comm: str, sugg: str) -> bool:
    if sugg: return True
    if not comm: return False
    if is_trivial(comm) or is_generic(comm): return False #
    if len(comm) <= LENGTH_THRESHOLD and not ( #
        has_technical_reference(comm) or is_suggestion(comm) #
        or is_design(comm)): #
        return False
    return True

# SUBDIR token
def subdir_token(path: str) -> str:
    parts = Path(path).parts[:2]
    return f"SUBDIR:{'/'.join(parts)}" if parts else ""

# Build/update index
def flush(ids: List[str], docs: List[str], metas: List[Dict[str, Any]]):
    if not ids: return
    # Ensure 'col' is globally available and initialized
    if col is None:
        print("[ERROR] ChromaDB collection 'col' is not initialized. Cannot flush.")
        # Optionally, raise an error or handle more gracefully
        raise RuntimeError("ChromaDB collection not initialized before flush.")
    
    vecs = embed(docs)
    col.add(ids=ids, embeddings=[v.tolist() for v in vecs],
            documents=docs, metadatas=metas)
    ids.clear(); docs.clear(); metas.clear()

def stream_json(fp: str):
    with open(fp, encoding="utf-8") as f:
        yield from ijson.items(f, "item")

def build_or_update():
    global client, col
    
    # Ensure persist_dir exists for PersistentClient
    if args.rebuild and os.path.exists(args.persist_dir):
        print(f"[INFO] Clearing directory {args.persist_dir}")
        shutil.rmtree(args.persist_dir, ignore_errors=True)
    
    Path(args.persist_dir).mkdir(parents=True, exist_ok=True)
    
    client = PersistentClient(path=args.persist_dir)
    col = client.get_or_create_collection("gcbert_embeds")

    # If not rebuilding or updating, we are done with the data processing part of setup.
    # This is the key change for faster startup in query-only mode.
    if not args.rebuild and not args.update:
        print(f"[INFO] ChromaDB collection '{col.name}' loaded with {col.count():,} items. Ready for queries.")
        return

    # --- The following logic is for --rebuild or --update scenarios ---
    existing_ids = set()
    if args.update: # Only fetch existing IDs if explicitly updating
        if col.count() > 0:
            print("[INFO] Fetching existing IDs from ChromaDB for update...")
            try:
                chroma_get_result = col.get(include=[]) # Fetches minimal data; IDs are always included
                if chroma_get_result and "ids" in chroma_get_result:
                    existing_ids = set(chroma_get_result["ids"])
                print(f"[INFO] Found {len(existing_ids)} existing IDs in ChromaDB.")
            except Exception as e:
                print(f"[WARN] Could not fetch existing IDs from ChromaDB: {e}")
        else:
            print("[INFO] ChromaDB collection is empty. Update will proceed by adding all items from the JSON file.")
            
    ids_to_add, docs_to_add, metas_to_add, seen_in_current_scan = [], [], [], set()
    
    action = "rebuild" if args.rebuild else "update"
    print(f"[INFO] Starting data processing from {args.data_file} for {action}...")
    
    for rec in tqdm.tqdm(stream_json(args.data_file), unit="rec"):
        diff = rec.get("diff_text", "")
        sugg = rec.get("suggestion_text", "")
        comm = rec.get("comment_text", "")
        
        if not keep_record(comm, sugg): 
            continue
        
        fn  = rec.get("filename", "")
        ast_text = "" 
        if CLANG_AVAILABLE and fn.endswith((".c", ".cpp", ".h", ".hpp", ".cc", ".cxx")):
            ast_text = parse_cpp_ast(diff)
            
        body = sugg if sugg else comm
        doc  = f"{subdir_token(fn)}\n{diff}\n{ast_text}\n{body}".strip()
        current_id  = md5(doc)

        # Avoid re-processing (and re-embedding) the same doc if it appears multiple times in the JSON file
        if current_id in seen_in_current_scan:
            continue
        
        if args.update and current_id in existing_ids:
            seen_in_current_scan.add(current_id) 
            continue

        ids_to_add.append(current_id)
        docs_to_add.append(doc)
        metas_to_add.append({
            "filename": fn,
            "commenter": rec.get("commenter", ""),
            "comment_text": comm,
            "suggestion_text": sugg
        })
        seen_in_current_scan.add(current_id)
        
        if len(ids_to_add) >= args.batch:
            flush(ids_to_add, docs_to_add, metas_to_add)
            # seen_in_current_scan is NOT cleared here. It tracks all unique items encountered 
            # in this entire JSON scan to avoid redundant embedding if the JSON has duplicates.
            # ChromaDB's `add` operation with IDs is idempotent, so sending the same ID multiple
            # times (if it happened due to clearing seen_in_current_scan) would be an update, not an error.
            # However, avoiding the `embed()` call for true duplicates in the JSON is the goal here.

    flush(ids_to_add, docs_to_add, metas_to_add)
    print(f"[INFO] Index {action} complete. Index size: {col.count():,}")

# Retrieval (use raw distances and compute similarity)
def retrieve(diff: str, k: int, fname: str | None = None):
    # Ensure 'col' is globally available and initialized
    if col is None:
        print("[ERROR] ChromaDB collection 'col' is not initialized. Cannot retrieve.")
        # Optionally, raise an error or handle more gracefully
        raise RuntimeError("ChromaDB collection not initialized before retrieval.")

    sub = subdir_token(fname) if fname else ""
    ast_text = parse_cpp_ast(diff) if CLANG_AVAILABLE else "" # Use ast_text
    qtext = f"{sub}\n{diff}\n{ast_text}".strip() # Use ast_text
    qvec  = embed(qtext)[0].tolist()
    res   = col.query(query_embeddings=[qvec], n_results=k,
                    include=["distances","metadatas"])
    dists = res["distances"][0]
    metas = res["metadatas"][0]
    return list(zip(dists, metas))

# Interactive
def best_text(meta: Dict[str, Any]) -> str:
    return meta.get("suggestion_text") or meta.get("comment_text", "")

def read_filename() -> str | None:
    fn = input("File path (optional, press enter to skip): ").strip()
    return fn or None

def read_diff() -> str:
    print("Paste diff (end with line 'EOF'):")
    buf = []
    while True:
        try:
            ln = input()
        except EOFError: # Handle Ctrl+D as EOF
            break
        if ln.strip().upper() == "EOF":
            break
        buf.append(ln)
    return "\n".join(buf)

def main():
    build_or_update() # This will now be fast if not rebuilding or updating
    print(f"\n[INFO] Ready — top-{args.top_k}, P(keep) ≥ {args.threshold}\n")
    while True:
        fname = read_filename()
        diff  = read_diff()
        if not diff.strip(): 
            print("Empty diff entered. Exiting.")
            break
        
        shown = 0
        try:
            results = retrieve(diff, args.top_k * 4, fname) # Retrieve more to filter down
        except RuntimeError as e: # Catch if col was not initialized
            print(f"[ERROR] Could not perform retrieval: {e}")
            break
        
        for dist, meta in results:
            text = best_text(meta)
            # Ensure keep_record checks against suggestion_text for its specific logic path
            if not text or not keep_record(meta.get("comment_text",""), meta.get("suggestion_text", "")): continue
            
            pk_val = p_keep(text)[0] # Use pk_val to avoid recomputing
            if pk_val < args.threshold: continue
            
            shown += 1
            sim = 1.0 - dist
            ttype = "suggestion" if meta.get("suggestion_text") else "comment"
            print(f"--- Recommendation #{shown} ---")
            print(f"  File       : {meta['filename']}")
            print(f"  Type       : {ttype}")
            print(f"  Distance   : {dist:.4f}")
            print(f"  Similarity : {sim:.4f}")
            print(f"  P(keep)    : {pk_val:.4f}")
            print("  Text:")
            for line in text.strip().splitlines():
                print(f"    {line}")
            print()
            if shown >= args.top_k:
                break
        
        if shown == 0:
            print("No recommendations found meeting the criteria for the given diff.")
            
        if not input("Try another diff? (y/n): ").lower().startswith('y'):
            break
    print("GG!")

if __name__ == "__main__":
    main()