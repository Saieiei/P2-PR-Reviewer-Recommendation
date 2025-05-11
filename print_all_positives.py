# export_all_positives.py
import pandas as pd
import csv
import sys
from pathlib import Path

# 1) Load your train split
path = Path(__file__).parent / "train.tsv"
if not path.exists():
    print(f"Error: {path} not found", file=sys.stderr)
    sys.exit(1)

cols = ["label", "pr_number", "commenter", "diff", "comment"]
df = pd.read_csv(
    path,
    sep="<CODESPLIT>",
    engine="python",
    header=None,
    names=cols,
    quoting=csv.QUOTE_NONE,
    keep_default_na=False
)

# 2) Coerce & filter positives
df["label"] = pd.to_numeric(df["label"], errors="coerce").fillna(0).astype(int)
positives = df[(df.label == 1) & (df.comment.str.strip().astype(bool))]

# 3) Write them all out
out_path = Path("all_positives.txt")
with open(out_path, "w", encoding="utf-8") as f:
    for i, txt in enumerate(positives.comment, start=1):
        f.write(f"{i}. {txt.strip()}\n\n")

print(f"Wrote {len(positives)} positive comments to {out_path.resolve()}")
