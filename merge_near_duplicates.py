#!/usr/bin/env python3
# merge_near_duplicates.py

import re
import pandas as pd

def normalize(s: str) -> str:
    """
    Lowercase, strip leading/trailing whitespace,
    remove punctuation, collapse multiple spaces.
    """
    s = s.lower().strip()
    # drop all non-alphanum and non-space
    s = re.sub(r'[^a-z0-9\s]', '', s)
    # collapse spaces
    s = re.sub(r'\s+', ' ', s)
    return s

def main():
    # 1) read the counts you generated before
    df = pd.read_excel("trivial_generic_comment_counts.xlsx")

    # 2) compute normalized key
    df["norm"] = df["comment"].astype(str).map(normalize)

    # 3) group by that key
    grouped = (
        df
        .groupby("norm", sort=False)
        .agg(
            total_count=pd.NamedAgg(column="count", aggfunc="sum"),
            variants=pd.NamedAgg(column="comment", aggfunc=lambda vs: list(vs))
        )
        .reset_index()
        .sort_values("total_count", ascending=False)
    )

    # 4) save
    out = "merged_comment_counts.xlsx"
    grouped.to_excel(out, index=False)
    print(f"✔ Merged into {len(grouped)} groups; saved to '{out}'")

if __name__ == "__main__":
    main()

