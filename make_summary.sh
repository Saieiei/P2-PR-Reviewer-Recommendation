#!/usr/bin/env bash
# coding=utf-8
#
# rewrite sweep_summary.csv to include acc, acc_and_f1 and f1

SWEEP_DIR="/ptmp2/moresair/cloned_repo/P2-PR-Reviewer-Recommendation/sweep_results"
OUT_CSV="${SWEEP_DIR}/sweep_summary.csv"

echo "exp_name,acc,acc_and_f1,f1" > "${OUT_CSV}"

for d in "${SWEEP_DIR}"/lr*; do
  if [[ -d "$d" && -f "$d/eval_results.txt" ]]; then
    name=$(basename "$d")
    ACC=$(grep '^acc '        "$d/eval_results.txt" | tail -1 | awk -F'= ' '{print $2}')
    AF1=$(grep '^acc_and_f1 ' "$d/eval_results.txt" | tail -1 | awk -F'= ' '{print $2}')
    F1=$(grep '^f1 '          "$d/eval_results.txt" | tail -1 | awk -F'= ' '{print $2}')
    echo "${name},${ACC},${AF1},${F1}" >> "${OUT_CSV}"
  fi
done

echo "✅ Rewritten ${OUT_CSV}"
