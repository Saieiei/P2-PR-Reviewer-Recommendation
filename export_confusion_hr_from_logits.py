#!/usr/bin/env python3
import os, re, math, argparse, sys

# ─── CONFIG ─────────────────────────────────────────────────────
CODESPLIT = "<CODESPLIT>"
OUT_DIR   = "/ptmp2/moresair/cloned_repo/P2-PR-Reviewer-Recommendation/debugging_files"
os.makedirs(OUT_DIR, exist_ok=True)

# ─── HELPERS ────────────────────────────────────────────────────
def softmax(l0, l1):
    e0, e1 = math.exp(l0), math.exp(l1)
    return e1 / (e0 + e1)

DIV = "=" * 60
SEP = "-" * 60

def pretty_write(fp, idx, diff, comment, true_lbl, pred_lbl, prob):
    # extract file, label, then the rest of the diff payload
    m = re.match(
        r"\[FILE\](?P<file>.*?)\s+\[LABEL\](?P<label>.*?)\s+(?P<payload>.*)",
        diff,
        flags=re.DOTALL
    )
    if m:
        file_path = m.group("file").strip()
        label_txt = m.group("label").strip()
        payload   = m.group("payload").rstrip()
    else:
        # fallback if prefix isn’t present
        file_path = ""
        label_txt = ""
        payload   = diff.strip()

    # header
    fp.write(f"{DIV}\n")
    fp.write(f"Example #{idx}\n")
    fp.write(f"True   : {true_lbl}\n")
    fp.write(f"Pred   : {pred_lbl}   (p_pos={prob:.3f})\n\n")

    # FILE & LABEL
    fp.write(f"FILE   : {file_path}\n")
    fp.write(f"LABEL  : {label_txt}\n\n")

    # DIFF section
    fp.write(f"{SEP} DIFF\n")
    fp.write(payload + "\n\n")

    # COMMENT section
    fp.write(f"{SEP} COMMENT\n")
    fp.write(comment.strip() + "\n")
    fp.write(f"{DIV}\n\n")

# ─── MAIN ───────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(
        description="Dump human-readable fp/fn/tp/tn from dev_logits.tsv"
    )
    p.add_argument(
        "--logits", required=True,
        help="Path to dev_logits.tsv"
    )
    p.add_argument(
        "--thr", type=float, default=0.47,
        help="Probability threshold for positive class"
    )
    args = p.parse_args()

    print(f"\n→ exporting `{args.logits}` with threshold={args.thr}\n", flush=True)

    # open files
    files = {
        "fp": open(os.path.join(OUT_DIR, "fp.txt"), "w"),
        "fn": open(os.path.join(OUT_DIR, "fn.txt"), "w"),
        "tp": open(os.path.join(OUT_DIR, "tp.txt"), "w"),
        "tn": open(os.path.join(OUT_DIR, "tn.txt"), "w"),
    }
    counts = {k: 0 for k in files}

    # process each line
    with open(args.logits, "r") as f:
        for idx, line in enumerate(f, start=1):
            parts = line.rstrip("\n").split(CODESPLIT)
            if len(parts) < 7:
                sys.exit(f"✖ Line #{idx} malformed: needs ≥7 fields, got {len(parts)}")
            true_lbl = int(parts[0])
            diff     = parts[3]
            comment  = parts[4]
            l0, l1   = float(parts[5]), float(parts[6])

            prob_pos = softmax(l0, l1)
            pred     = int(prob_pos >= args.thr)

            key = (
                "fp" if (true_lbl, pred) == (0,1) else
                "fn" if (true_lbl, pred) == (1,0) else
                "tp" if (true_lbl, pred) == (1,1) else
                "tn"
            )
            pretty_write(files[key], idx, diff, comment, true_lbl, pred, prob_pos)
            counts[key] += 1

            # progress every 500
            if idx % 500 == 0:
                print(f"[{idx} lines]  FP={counts['fp']}  FN={counts['fn']}  TP={counts['tp']}  TN={counts['tn']}", flush=True)

    # close & summary
    for fh in files.values():
        fh.close()

    total = sum(counts.values())
    print(f"\n✔ Done ({total} examples)")
    for k in ("fp","fn","tp","tn"):
        print(f"  {k.upper():>2}: {counts[k]}")
    print(f"\nFiles written to {OUT_DIR}\n")

if __name__ == "__main__":
    main()
