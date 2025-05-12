#!/usr/bin/env python3
import random
import re
from rapidfuzz import fuzz
from tqdm import tqdm
from process_data_pr import normalize, GENERIC_EXAMPLES

# Precompute normalized patterns
GENERIC_NORMS   = [ normalize(p) for p in GENERIC_EXAMPLES ]
FUZZY_THRESHOLD = 92
TYPO_THRESHOLD  = 80

def max_score(normed: str) -> int:
    return max(fuzz.ratio(normed, pat) for pat in GENERIC_NORMS)

def is_generic_fast(normed: str) -> bool:
    if normed in GENERIC_NORMS:
        return True
    score = max_score(normed)
    max_pat_len = max(len(p) for p in GENERIC_NORMS)
    delta = max(2, min(4, int(max_pat_len * 0.2)))
    if any(abs(len(normed) - len(p)) <= delta for p in GENERIC_NORMS) and score >= TYPO_THRESHOLD:
        return True
    return score >= FUZZY_THRESHOLD

def main():
    # 1) load raw comments
    with open("all_comments.txt", encoding="utf-8") as f:
        raw = [line.rstrip() for line in f if line.strip()]

    # 2) normalize, but keep pairs and drop empty normals
    pairs = []
    for orig in raw:
        n = normalize(orig)
        if len(n) >= 3:      # drop too-short or empty
            pairs.append((orig, n))

    # 3) scan with progress bar
    results = []
    for orig, normed in tqdm(pairs, desc="Auditing comments"):
        if is_generic_fast(normed):
            sc = max_score(normed)
            results.append((orig, sc))

    # 4) inspect per threshold
    for T in [93, 90, 85, 80]:
        flt = [(o, sc) for o, sc in results if sc >= T]
        print(f"\nThreshold ≥{T}: {len(flt)} comments filtered out")
        for orig, sc in random.sample(flt, min(10, len(flt))):
            print(f"  [{int(sc):2d}] {orig}")

if __name__ == "__main__":
    main()
