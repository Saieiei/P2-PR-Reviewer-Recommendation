#!/usr/bin/env python3
import math
import sys
from sklearn.metrics import confusion_matrix, classification_report

def softmax(logit0, logit1):
    m = max(logit0, logit1)
    e0 = math.exp(logit0 - m)
    e1 = math.exp(logit1 - m)
    s = e0 + e1
    return e1 / s

def main(dev_logits_path, thr=0.5, n_show=5):
    y_true, y_pred = [], []
    false_pos, false_neg = [], []

    with open(dev_logits_path, encoding='utf-8') as f:
        for i, line in enumerate(f):
            parts = line.rstrip('\n').split("<CODESPLIT>")
            # inst = parts[:-2], logits = parts[-2:]
            try:
                label = int(parts[0])
                l0, l1 = float(parts[-2]), float(parts[-1])
            except:
                print(f"skipping malformed line {i}", file=sys.stderr)
                continue

            p1 = softmax(l0, l1)
            pred = 1 if p1 > thr else 0

            y_true.append(label)
            y_pred.append(pred)

            if pred == 1 and label == 0 and len(false_pos) < n_show:
                false_pos.append(parts)   # store full instance for later
            if pred == 0 and label == 1 and len(false_neg) < n_show:
                false_neg.append(parts)

    # print overall metrics
    print("Confusion matrix:")
    print(confusion_matrix(y_true, y_pred))
    print()
    print("Classification report:")
    print(classification_report(y_true, y_pred, digits=4))
    print()

    # dump a few examples
    print(f"--- {len(false_pos)} false-positives (model said 1, truth 0) ---")
    for inst in false_pos:
        diff = inst[3]  # this is your diff window
        comment = inst[4]
        print(f"\nDIFF WINDOW:\n{diff}\nCOMMENT:\n{comment}\n")

    print(f"--- {len(false_neg)} false-negatives (model said 0, truth 1) ---")
    for inst in false_neg:
        diff = inst[3]
        comment = inst[4]
        print(f"\nDIFF WINDOW:\n{diff}\nCOMMENT:\n{comment}\n")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: analyze_errors_fixed.py dev_logits.tsv [threshold]", file=sys.stderr)
        sys.exit(1)
    path = sys.argv[1]
    thr = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
    main(path, thr)
