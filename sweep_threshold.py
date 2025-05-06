import numpy as np
from sklearn.metrics import f1_score
import sys

# Read dev_logits.tsv
# Each line: <label><CODESPLIT><logit_neg><CODESPLIT><logit_pos>
labels, logits = [], []
with open("dev_logits.tsv") as f:
    for line in f:
        parts = line.strip().split("<CODESPLIT>")
        label = int(parts[0])
        logit_pos = float(parts[-1])  # last column
        labels.append(label)
        logits.append(logit_pos)

probs = 1 / (1 + np.exp(-np.array(logits)))  # sigmoid

best_thr, best_f1 = 0.0, 0.0
for thr in np.linspace(0, 1, 101):
    preds = (probs > thr).astype(int)
    f1 = f1_score(labels, preds)
    if f1 > best_f1:
        best_f1, best_thr = f1, thr

print(f"Best dev F₁ = {best_f1:.4f} at threshold = {best_thr:.2f}")

