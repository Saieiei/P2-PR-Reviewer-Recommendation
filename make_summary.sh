#!/usr/bin/env bash
# coding=utf-8
#
# Rewrite sweep_summary.csv to include acc, precision, recall, acc_and_f1 and f1
# For each run we keep the *highest* value seen across all epochs.

###############################################################################
# CONFIG
###############################################################################
SWEEP_DIR="/ptmp2/moresair/cloned_repo/P2-PR-Reviewer-Recommendation/sweep_results/graphcodebert"
OUT_CSV="${SWEEP_DIR}/sweep_summary.csv"
###############################################################################

shopt -s nullglob          # ignore empty globs

# helper: read values from stdin → emit the numeric max
best() { awk -F'= ' '{print $2}' | sort -gr | head -1; }

# header
echo "exp_name,acc,precision,recall,acc_and_f1,f1" > "$OUT_CSV"

# iterate over each experiment directory
for d in "$SWEEP_DIR"/*; do
  [[ -f "$d/eval_results.txt" ]] || continue

  ACC=$(grep   '^acc '         "$d/eval_results.txt" | best)
  PREC=$(grep  '^precision '   "$d/eval_results.txt" | best)
  REC=$(grep   '^recall '      "$d/eval_results.txt" | best)
  AF1=$(grep   '^acc_and_f1 '  "$d/eval_results.txt" | best)
  F1=$(grep    '^f1 '          "$d/eval_results.txt" | best)

  echo "$(basename "$d"),$ACC,$PREC,$REC,$AF1,$F1" >> "$OUT_CSV"
done

rows=$(($(wc -l < "$OUT_CSV") - 1))
echo "✅ Rewritten ${OUT_CSV}  (${rows} rows)"
