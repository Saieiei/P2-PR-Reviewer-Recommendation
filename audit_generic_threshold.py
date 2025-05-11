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

def max_score(c: str) -> int:
    """Return the highest fuzz.ratio(c, pattern) over all patterns."""
    return max(fuzz.ratio(c, pat) for pat in GENERIC_NORMS)

def is_generic_fast(c: str) -> bool:
    """
    1) exact match
    2) short‐comment typo‐mode (±delta length → TYPO_THRESHOLD)
    3) strict‐mode for longer comments → FUZZY_THRESHOLD
    """
    # exact-match catch
    if c in GENERIC_NORMS:
        return True

    score = max_score(c)
    # adaptive delta based on longest pattern
    max_pat_len = max(len(p) for p in GENERIC_NORMS)
    delta = max(2, min(4, int(max_pat_len * 0.2)))

    # typo-mode for any pattern-length neighbor
    if any(abs(len(c) - len(p)) <= delta for p in GENERIC_NORMS) and score >= TYPO_THRESHOLD:
        return True

    # strict catch for nearly identical long strings
    return score >= FUZZY_THRESHOLD

def main():
    # 1) load comments
    with open("all_comments.txt", encoding="utf-8") as f:
        comments = [line.rstrip() for line in f if line.strip()]

    # 2) normalize once
    normed = [normalize(c) for c in comments]

    # 3) scan with progress bar
    results = []
    for c in tqdm(normed, desc="Auditing comments"):
        if is_generic_fast(c):
            sc = max_score(c)
            results.append((c, sc))

    # 4) inspect per threshold
    for T in [93, 90, 85, 80]:
        flt = [ (c,sc) for c,sc in results if sc >= T ]
        print(f"\nThreshold ≥{T}: {len(flt)} comments filtered out")
        for c, sc in random.sample(flt, min(10, len(flt))):
            print(f"  [{int(sc):2d}] {c}")

if __name__ == "__main__":
    main()
