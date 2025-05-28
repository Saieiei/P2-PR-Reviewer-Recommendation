#!/usr/bin/env python3
# retrieval_gcbert_by_prid.py
"""
Given a GitHub PR number, print the top-N GC-BERT suggestions for **every**
diff-hunk of every file touched in that PR.

Requires:
  • pull_requests.db          (schema from collect_pr_data.py)
  • retrieval_gcbert.py       (in the same directory, index already built
                               or buildable on the fly)
"""

from __future__ import annotations
import argparse, sys, sqlite3, math, textwrap
from pathlib import Path

# ───────────────────────────────────
# 0) Pre-parse wrapper-only flags
#    (remove them from sys.argv)
# ───────────────────────────────────
wrapper_parser = argparse.ArgumentParser(add_help=False) #
wrapper_parser.add_argument("--db", default="pull_requests.db", #
                            help="SQLite DB from collect_pr_data.py") #
wrapper_parser.add_argument("--top-k-hunks", type=int, default=10, #
                            help="Top suggestions *per hunk* (default 10)") #
wrapper_args, remaining_argv = wrapper_parser.parse_known_args() #

# Keep only the args that retrieval_gcbert knows
sys.argv = [sys.argv[0]] + remaining_argv #

# ───────────────────────────────────
# 1)  Import GC-BERT retrieval stack
#     (its arg-parser runs on import)
# ───────────────────────────────────
import retrieval_gcbert as rg          # noqa: E402 #

# ───────────────────────────────────
# 2)  Small helpers
# ───────────────────────────────────
def split_hunks(diff: str) -> list[str]: #
    """Return each @@ … @@ hunk as a separate string (header line included)."""
    hunks, cur = [], [] #
    for ln in diff.splitlines(): #
        if ln.startswith("@@"): #
            if cur: #
                hunks.append("\n".join(cur)) #
            cur = [ln] #
        else: #
            cur.append(ln) #
    if cur: #
        hunks.append("\n".join(cur)) #
    return [h for h in hunks if h.strip()] #

def canon(txt: str) -> str: #
    return " ".join(txt.split()).lower() #

def heading(txt: str) -> None: #
    bar = "=" * len(txt) #
    print(f"{bar}\n{txt}\n{bar}") #

# ───────────────────────────────────
# 3)  Suggestion printer
# ───────────────────────────────────
def show_suggestions(fn: str, hunk: str, top_k: int) -> None: #
    # rg.retrieve now uses a cosine-based collection.
    # dist_sq will be 1 - cosine_similarity (small for good matches)
    res  = rg.retrieve(hunk, top_k * 10, fn)  # over-fetch → survives dedup #
    seen = set() #
    shown = 0 #

    for dist_sq, meta in res: #
        txt = rg.best_text(meta) #
        if not txt: #
            continue
        if canon(txt) in seen: #
            continue
        if not rg.keep_record(meta.get("comment_text", ""), #
                              meta.get("suggestion_text", "")): #
            continue
        pk = rg.p_keep(txt)[0] #
        if pk < rg.args.threshold: #
            continue

        seen.add(canon(txt)) #
        shown += 1 #

        # dist_sq is 1 - cosine_similarity (e.g., 0.01 for high similarity)
        l2   = math.sqrt(max(dist_sq, 0.0)) # e.g., sqrt(0.01) = 0.1 #
        # sim will be high for good matches, e.g., 1 / (1+0.1) ~ 0.909
        sim  = 1.0 / (1.0 + l2) #
        kind = "suggestion" if meta.get("suggestion_text") else "comment" #

        # *** PATCHED LINE: Print dist_sq (1-cosine_similarity) as "dist" ***
        print(f"    {shown:2d}. ({kind}, dist={dist_sq:.3f}, sim={sim:.3f}, P={pk:.3f})") #
        # Was: print(f"    {shown:2d}. ({kind}, dist={1-sim:.3f}, sim={sim:.3f}, P={pk:.3f})")

        for line in txt.strip().splitlines(): #
            print(f"        {line}") #
        if shown >= top_k: #
            break

    if shown == 0: #
        print("    (no suggestions passed the filters)") #

# ───────────────────────────────────
# 4)  DB → file/hunk → GC-BERT loop
# ───────────────────────────────────
def process_pr(cur: sqlite3.Cursor, pr_number: int, top_k: int) -> None: #
    row = cur.execute( #
        "SELECT id FROM PullRequests WHERE pr_number = ?", (pr_number,) #
    ).fetchone() #
    if not row: #
        print(f"❌  PR #{pr_number} not found in DB") #
        return
    pr_id = row[0] #

    for fn, diff in cur.execute( #
        "SELECT filename, diff_text FROM PRFiles WHERE pr_id = ?", (pr_id,) #
    ):
        if not diff.strip(): #
            continue
        hunks = split_hunks(diff) or [diff] #
        for idx, hunk in enumerate(hunks, 1): #
            heading(f"{fn}  —  hunk {idx}/{len(hunks)}") #
            # optional preview of the diff (first ~20 lines)
            preview = "\n".join(hunk.strip().splitlines()[:20]) #
            print(textwrap.indent(preview, ">> "))  # comment out if too noisy #
            print() #
            show_suggestions(fn, hunk, top_k) #
            print() #

# ───────────────────────────────────
# 5)  CLI & main loop
# ───────────────────────────────────
def main() -> None: #
    # ap = argparse.ArgumentParser( # This was simplified in the original, keep it that way
    #     description="GC-BERT suggestions for every hunk in a PR",
    #     add_help=True,
    # )
    # ap.add_argument("--db", default="pull_requests.db", # Handled by wrapper_args
    #                 help="SQLite DB from collect_pr_data.py")
    # ap.add_argument("--top-k", type=int, default=10,     # Handled by wrapper_args
    #                 help="Top suggestions per hunk (default 10)")
    # args, _unknown = ap.parse_known_args()

    # ensure GC-BERT index is ready (will build if needed using cosine)
    rg.build_or_update() #

    with sqlite3.connect(wrapper_args.db) as conn: #
        cur = conn.cursor() #
        while True: #
            try:
                entry = input("PR-ID (blank to quit): ").strip() #
            except EOFError: #
                print() #
                break
            if not entry: #
                break
            if not entry.isdigit(): #
                print("Please enter a numeric PR-ID.") #
                continue
            process_pr(cur, int(entry), wrapper_args.top_k_hunks) #

    print("Done – bye!") #

# ───────────────────────────────────
if __name__ == "__main__":
    main() #