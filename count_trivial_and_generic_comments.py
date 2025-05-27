#!/usr/bin/env python3
# count_trivial_and_generic_comments.py

import os
import sys
import json
from collections import Counter

import pandas as pd
from tqdm import tqdm

# import your existing filters
import process_data_pr as pdp

def main():
    DATA_FILE = "preprocessed_data.json"
    if not os.path.exists(DATA_FILE):
        print(f"❌ Error: '{DATA_FILE}' not found. Run process_data_pr.py first.")
        sys.exit(1)

    # ─── Load ───────────────────────────────────────────────────────────
    print("▶️  Loading preprocessed_data.json…")
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        records = json.load(f)
    total = len(records)
    print(f"   ✓ Loaded {total} records.\n")

    # ─── Filter ─────────────────────────────────────────────────────────
    print("▶️  Scanning for trivial/generic comments…")
    filtered_comments = []
    for rec in tqdm(records, desc="Records", unit="rec"):
        txt = rec.get("comment_text", "").strip()
        if not txt:
            continue
        if pdp.is_trivial(txt) or pdp.is_generic(txt):
            filtered_comments.append(txt)

    print(f"   ✓ Found {len(filtered_comments)} total trivial/generic comments.\n")

    # ─── Count ──────────────────────────────────────────────────────────
    print("▶️  Counting frequencies…")
    counts = Counter(filtered_comments)

    # ─── Save ───────────────────────────────────────────────────────────
    df = pd.DataFrame(
        sorted(counts.items(), key=lambda x: x[1], reverse=True),
        columns=["comment", "count"]
    )
    out_path = "trivial_generic_comment_counts.xlsx"
    df.to_excel(out_path, index=False)
    print(f"   ✓ {len(counts)} unique strings. Saved to '{out_path}'.\n")

if __name__ == "__main__":
    main()
