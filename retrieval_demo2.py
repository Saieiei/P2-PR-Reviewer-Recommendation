#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import json
import numpy as np
import torch

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise      import linear_kernel
from transformers                   import RobertaTokenizer, RobertaForSequenceClassification

# ----------------------------------------------------------------------
# CONFIGURATION
# ----------------------------------------------------------------------
DATA_FILE       = "preprocessed_data.json"
TOP_K           = 5
CLS_MODEL_DIR   = "./codebert-finetuned/checkpoint-best"
CLS_TOKENIZER   = "microsoft/codebert-base"
CLS_MAX_SEQ_LEN = 200
CLS_THRESHOLD   = 0.51

# ----------------------------------------------------------------------
# 1) TF-IDF INDEX
# ----------------------------------------------------------------------
print("Loading TF-IDF corpus...", file=sys.stderr)
with open(DATA_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

diff_texts    = [r["diff_text"]    for r in data]
comment_texts = [r["comment_text"] for r in data]

vectorizer  = TfidfVectorizer(
    stop_words="english",
    max_features=50000,
    ngram_range=(1,2)
)
diff_matrix = vectorizer.fit_transform(diff_texts)
print(
    f"Indexed {diff_matrix.shape[0]} diffs, TF-IDF dim={diff_matrix.shape[1]}",
    file=sys.stderr
)

def retrieve_tfidf(query, k=TOP_K):
    qv   = vectorizer.transform([query])
    sims = linear_kernel(qv, diff_matrix).flatten()
    idxs = np.argpartition(sims, -k)[-k:]
    idxs = idxs[np.argsort(-sims[idxs])]
    return [
        (i, float(sims[i]), comment_texts[i], diff_texts[i])
        for i in idxs
    ]

# ----------------------------------------------------------------------
# 2) LOAD THE CLASSIFIER
# ----------------------------------------------------------------------
device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Loading classifier on {device}", file=sys.stderr)

tok_cls   = RobertaTokenizer.from_pretrained(CLS_TOKENIZER)
model_cls = RobertaForSequenceClassification.from_pretrained(CLS_MODEL_DIR)
model_cls.to(device)
model_cls.eval()

# ----------------------------------------------------------------------
# 3) INTERACTIVE LOOP
# ----------------------------------------------------------------------
print(
    "Paste your diff lines. Type EOF on its own line to finish, or Ctrl-D to exit.",
    file=sys.stderr
)
print(
    f"Showing up to {TOP_K} candidates with P(useful) >= {CLS_THRESHOLD:.2f}",
    file=sys.stderr
)

while True:
    print("\n=== New query ===", file=sys.stderr)
    lines = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        if line.strip() == "EOF":
            break
        lines.append(line)
    if not lines:
        print("Empty input, exiting.", file=sys.stderr)
        break

    query = "\n".join(lines)

    # retrieve
    candidates = retrieve_tfidf(query, TOP_K)

    # classify & display
    shown = False
    for rank, (idx, sim, comment, past_diff) in enumerate(candidates, start=1):
        encoding = tok_cls(
            past_diff,
            comment,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=CLS_MAX_SEQ_LEN
        ).to(device)

        with torch.no_grad():
            logits = model_cls(**encoding).logits
            probs  = torch.softmax(logits, dim=-1)
        p_pos = probs[0,1].item()

        if p_pos < CLS_THRESHOLD:
            continue

        shown = True
        print(f"\nCandidate #{rank}: TF-IDF sim={sim:.4f}, P(useful)={p_pos:.3f}")
        print("Comment:")
        for ln in comment.splitlines():
            print("  " + ln)

    if not shown:
        print("No candidates above threshold.", file=sys.stderr)

print("Goodbye!", file=sys.stderr)
