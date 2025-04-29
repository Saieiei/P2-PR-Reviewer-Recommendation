import json
import random
import re
import os
from typing import List, Tuple

# === Configuration ===
DATA_FILE        = "preprocessed_data.json"
OUTPUT_DIR       = "."              # where train.tsv, dev.tsv, test.tsv go
CODESPLIT        = "<CODESPLIT>"     # separator for run_classifier.py
TRAIN_RATIO      = 0.80
DEV_RATIO        = 0.10
SEED             = 42

# Augmentation settings
WINDOW_SIZE      = 200              # tokens per window
OVERLAP          = 50               # tokens overlap between windows
SUGG_OVERSAMPLE  = 2                # how many times to duplicate suggestion positives

# Trivial-comment patterns (from prep_data_clean.py)
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

def is_trivial(text: str) -> bool:
    txt = (text or "").strip()
    if len(txt) < 5:
        return True
    for pat in _TRIVIAL_RE:
        if pat.match(txt):
            return True
    if not re.search(r"[A-Za-z0-9]", txt):
        return True
    return False

def sliding_windows(diff: str) -> List[str]:
    """Break a diff into overlapping windows of WINDOW_SIZE tokens."""
    tokens = diff.split()
    if len(tokens) <= WINDOW_SIZE:
        return [" ".join(tokens)]
    windows = []
    step = WINDOW_SIZE - OVERLAP
    for start in range(0, len(tokens), step):
        end = start + WINDOW_SIZE
        chunk = tokens[start:end]
        windows.append(" ".join(chunk))
        if end >= len(tokens):
            break
    return windows

# === Load records ===
with open(DATA_FILE, "r", encoding="utf-8") as f:
    records = json.load(f)

# === Build positives ===
# Each: (label=1, pr_number, commenter, diff_window, paired_text)
positives: List[Tuple[int,str,str,str,str]] = []
for rec in records:
    pr       = str(rec.get("pr_number","")).strip()
    filename = rec.get("filename","").strip()
    labels   = rec.get("labels","").strip()
    raw_diff = rec.get("diff_text","").strip()
    sugg     = rec.get("suggestion_text","").strip()
    comm     = rec.get("comment_text","").strip()
    commenter= rec.get("commenter","").strip()

    if not raw_diff:
        continue

    # metadata injection prefix
    prefix = f"[FILE]{filename} [LABEL]{labels} "

    # generate windows
    for win in sliding_windows(raw_diff):
        diff_win = prefix + win

        # suggestion positives (oversampled)
        if sugg:
            for _ in range(SUGG_OVERSAMPLE):
                positives.append((1, pr, commenter, diff_win, sugg))

        # comment positives
        if comm and comm != sugg and not is_trivial(comm):
            positives.append((1, pr, commenter, diff_win, comm))

print(f"> Created {len(positives)} positive examples (with augmentation & oversampling)")

# === Build negatives (1 per positive) ===
random.seed(SEED)
negatives: List[Tuple[int,str,str,str,str]] = []
idxs = list(range(len(positives)))
for i, (lbl, pr_a, commr_a, diff_a, text_a) in enumerate(positives):
    # pick a random positive from a different PR
    while True:
        j = random.choice(idxs)
        if j != i and positives[j][1] != pr_a:
            break
    _, pr_b, commr_b, _, text_b = positives[j]
    negatives.append((0, pr_a, commr_b, diff_a, text_b))

print(f"> Created {len(negatives)} negative examples")

# === Combine & shuffle ===
all_examples = positives + negatives
random.shuffle(all_examples)

# === Split ===
n_total = len(all_examples)
n_train = int(n_total * TRAIN_RATIO)
n_dev   = int(n_total * DEV_RATIO)

train_set = all_examples[:n_train]
dev_set   = all_examples[n_train:n_train + n_dev]
test_set  = all_examples[n_train + n_dev:]

print(f"> Train/dev/test counts: {len(train_set)}/{len(dev_set)}/{len(test_set)}")

# === Write TSV helper ===
def write_tsv(exs: List[Tuple[int,str,str,str,str]], path: str):
    with open(path, "w", encoding="utf-8") as out:
        for label, url_a, url_b, txt_a, txt_b in exs:
            a = txt_a.replace("\n", " ")
            b = txt_b.replace("\n", " ")
            line = f"{label}{CODESPLIT}{url_a}{CODESPLIT}{url_b}{CODESPLIT}{a}{CODESPLIT}{b}\n"
            out.write(line)

# === Emit files ===
os.makedirs(OUTPUT_DIR, exist_ok=True)
write_tsv(train_set, os.path.join(OUTPUT_DIR, "train.tsv"))
write_tsv(dev_set,   os.path.join(OUTPUT_DIR, "dev.tsv"))
write_tsv(test_set,  os.path.join(OUTPUT_DIR, "test.tsv"))

print("✔ process_data_pr.py complete — train/dev/test ready for run_classifier.py")

