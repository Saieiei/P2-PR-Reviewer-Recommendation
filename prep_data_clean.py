# -*- coding: utf-8 -*-
"""
prep_data.py  –  scrape + clean + keep suggested‑patches (with labels)

1. Read from pull_requests.db (PullRequests, PRFiles, PRComments).
2. For every *useful* review comment, emit a JSON record:
      diff_text, comment_text, suggestion_text, filename, commenter, labels, …
3. Trivial “LGTM / thanks / 👍” comments are skipped *unless* they contain a
   fenced GitHub *suggested‑change* block (```suggestion …```).

Run:
    python prep_data.py
"""

import json
import os
import re
import sqlite3
from typing import Dict, List

DB_NAME = "pull_requests.db"
OUTPUT_JSON = "preprocessed_data.json"
BATCH_SIZE = 1_000

# ------------------------------------------------------------------ #
# 1)  Helpers – trivial‑comment detection & suggestion extraction    #
# ------------------------------------------------------------------ #
_TRIVIAL_PATTERNS = [
    r"^\s*lgtm\s*[.!]*$",
    r"^\s*looks\s+good\s+to\s+me\s*[.!]*$",
    r"^\s*approved\s*[.!]*$",
    r"^\s*ship\s+it\s*[.!]*$",
    r"^\s*thanks[,!.]*\s*$",
    r"^\s*thank\s+you[,!.]*\s*$",
    r"^\s*done[,!.]*\s*$",
    r"^\s*nit\b.*$",
    r"^\s*\+1\s*$",
]
_TRIVIAL_RE = [re.compile(pat, re.I) for pat in _TRIVIAL_PATTERNS]

_SUGG_RE = re.compile(r"```suggestion.*?\n(.*?)```", re.S | re.I)


def extract_suggestion(text: str) -> str:
    """Return concatenated suggested‑patch blocks or ''."""
    blocks = _SUGG_RE.findall(text or "")
    return "\n\n".join(b.strip() for b in blocks)


def is_trivial(text: str) -> bool:
    """True if *text* is a low‑value one‑liner (emoji, thanks, LGTM …)."""
    txt = (text or "").strip()
    if len(txt) < 5 and not extract_suggestion(txt):
        return True
    for pat in _TRIVIAL_RE:
        if pat.match(txt):
            return True
    # All‑emoji / no alphanumerics
    if not re.search(r"[A-Za-z0-9]", txt):
        return True
    return False


# ------------------------------------------------------------------ #
# 2)  DB → cleaned JSON                                              #
# ------------------------------------------------------------------ #
def process_rows(rows, out: List[Dict], total: int, kept: int):
    for row in rows:
        total += 1
        if total % BATCH_SIZE == 0:
            print(f"Processed {total:,} rows; kept {kept:,} examples.")

        # Updated query: now each row includes p.labels as the 4th column.
        (
            pr_number,
            pr_title,
            pr_desc,
            pr_labels,    # new column from the PullRequests table
            filename,
            diff_text,
            commenter,
            comment_text,
            comment_file,
            comment_line,
            created_at,
        ) = row

        diff_text = (diff_text or "").strip()
        comment_text = (comment_text or "").strip()
        suggestion_text = extract_suggestion(comment_text)

        # Use default value "N/A" if pr_labels is empty
        labels = pr_labels if pr_labels and pr_labels.strip() != "" else "N/A"

        if not diff_text or (is_trivial(comment_text) and not suggestion_text):
            continue  # skip trivial

        out.append(
            {
                "pr_number": pr_number,
                "title": pr_title or "",
                "description": pr_desc or "",
                "labels": labels,      # added labels with default if empty
                "filename": filename or "",
                "diff_text": diff_text,
                "commenter": commenter or "",
                "comment_text": comment_text,
                "suggestion_text": suggestion_text,
                "comment_file_path": comment_file,
                "comment_line_num": comment_line,
                "comment_created_at": created_at,
            }
        )
        kept += 1
    return total, kept


def main():
    if not os.path.exists(DB_NAME):
        print(f"DB '{DB_NAME}' not found.")
        return

    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()

    # -------------------------------------------------------------------
    # Updated Query 1: File‑matched comments with labels from PullRequests.
    # -------------------------------------------------------------------
    q1 = """
    SELECT p.pr_number, p.title, p.description, p.labels,
           f.filename, f.diff_text,
           c.commenter, c.comment_text,
           c.file_path, c.line_number, c.created_at
    FROM PullRequests  p
    JOIN PRFiles       f ON p.id = f.pr_id
    JOIN PRComments    c ON c.pr_id = p.id AND c.file_path = f.filename
    """

    # -------------------------------------------------------------------
    # Updated Query 2: Orphan comments with labels.
    # -------------------------------------------------------------------
    q2 = """
    SELECT p.pr_number, p.title, p.description, p.labels,
           '', '',                   -- filename, diff_text
           c.commenter, c.comment_text,
           c.file_path, c.line_number, c.created_at
    FROM PullRequests p
    JOIN PRComments  c ON c.pr_id = p.id
    WHERE c.file_path IS NULL
    """

    examples: List[Dict] = []
    total = kept = 0
    print("Querying file‑matched comments …")
    total, kept = process_rows(cur.execute(q1), examples, total, kept)
    print("Querying orphan comments …")
    total, kept = process_rows(cur.execute(q2), examples, total, kept)
    conn.close()

    print("Writing JSON …")
    with open(OUTPUT_JSON, "w", encoding="utf-8") as fp:
        json.dump(examples, fp, ensure_ascii=False, indent=2)

    print(f"Done. Visited {total:,} DB rows, wrote {kept:,} examples → {OUTPUT_JSON}.")


if __name__ == "__main__":
    main()

