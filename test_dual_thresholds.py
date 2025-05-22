#!/usr/bin/env python3
import re
from rapidfuzz import fuzz

# — your “fixed” parameters —
PAT            = "fixed"
PAT_NORM       = re.sub(r"\W+", " ", PAT).lower().strip()
TYPO_THRESHOLD = 80
STRICT_THRESHOLD = 92
DELTA          = max(2, min(4, int(len(PAT_NORM) * 0.2)))  # = 2

def normalize(s: str) -> str:
    return re.sub(r"\W+", " ", s).lower().strip()

def is_generic_fixed(normed: str) -> bool:
    # 1) exact match
    if normed == PAT_NORM:
        return True
    # 2) typo‐mode for short comments
    score = fuzz.ratio(normed, PAT_NORM)
    if abs(len(normed) - len(PAT_NORM)) <= DELTA and score >= TYPO_THRESHOLD:
        return True
    # 3) strict‐mode for longer comments
    if len(normed) > len(PAT_NORM) + DELTA and score >= STRICT_THRESHOLD:
        return True
    return False

def main():
    # load all comments
    with open("all_comments.txt", encoding="utf-8") as f:
        raw = [line.rstrip() for line in f if line.strip()]

    # bucket by short/long qualification
    short_cands = []
    long_cands  = []
    for orig in raw:
        n = normalize(orig)
        # only test those that mention “fixed” OR are within ±DELTA of pattern length
        if ("fixed" in n.split()) or (abs(len(n) - len(PAT_NORM)) <= DELTA):
            if len(n) <= len(PAT_NORM) + DELTA:
                short_cands.append((orig, n))
            else:
                long_cands.append((orig, n))

    # apply filter
    dropped_short = [o for o,n in short_cands if is_generic_fixed(n)]
    kept_short   = [o for o,n in short_cands if not is_generic_fixed(n)]
    dropped_long  = [o for o,n in long_cands if is_generic_fixed(n)]
    kept_long    = [o for o,n in long_cands if not is_generic_fixed(n)]

    # print counts
    print(f"Short candidates (typo‐mode): {len(short_cands)}")
    print(f"  Dropped short: {len(dropped_short)}")
    print(f"  Kept   short: {len(kept_short)}\n")

    print(f"Long candidates (strict‐mode): {len(long_cands)}")
    print(f"  Dropped long: {len(dropped_long)}")
    print(f"  Kept   long: {len(kept_long)}\n")

    # print all dropped short
    print("=== ALL DROPPED SHORT COMMENTS ===\n")
    for c in dropped_short:
        print(c)
    print("\n=== END DROPPED SHORT ===\n")

    # print all kept short
    print("=== ALL KEPT SHORT COMMENTS ===\n")
    for c in kept_short:
        print(c)
    print("\n=== END KEPT SHORT ===\n")

    # print all dropped long (should be none)
    print("=== ALL DROPPED LONG COMMENTS ===\n")
    for c in dropped_long:
        print(c)
    print("\n=== END DROPPED LONG ===\n")

    # print all kept long
    print("=== ALL KEPT LONG COMMENTS ===\n")
    for c in kept_long:
        print(c)
    print("\n=== END KEPT LONG ===\n")

if __name__ == "__main__":
    main()
