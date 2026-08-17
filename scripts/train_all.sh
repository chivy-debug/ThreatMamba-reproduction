#!/usr/bin/env bash
# ==========================================================================
# Stages 5 + 6 - train ALL FOUR configurations with the SAME config and seed,
# then evaluate them.
#
# Usage:  bash scripts/train_all.sh
#         bash scripts/train_all.sh --epochs 60     # extra args are forwarded to src.train
#
# WHY THIS SCRIPT EXISTS: on the first run each ablation trained for a different number of
# epochs (main 7, no_cl 41), which made the comparison table meaningless. Running them from
# one script guarantees all four share one configuration, one seed and one stopping rule.
#
# Runtime: about four times a single training run. Add --epochs for a quick pass.
# ==========================================================================
set -uo pipefail
cd "$(dirname "$0")/.."

if [ -t 1 ]; then G=$'\e[32m'; R=$'\e[31m'; B=$'\e[1m'; N=$'\e[0m'
else G=""; R=""; B=""; N=""; fi

[ -f .venv/bin/activate ] && . .venv/bin/activate

if [ ! -f data/processed/docs.jsonl ]; then
  echo "${R}data/processed/docs.jsonl is missing - run Stage 1 first.${N}"; exit 1
fi
if [ -z "$(ls -A data/cskg 2>/dev/null)" ]; then
  echo "${R}No CSKGs in data/cskg - run Stage 4 first: python -m src.cskg_builder build${N}"; exit 1
fi

FLAGS=("" "--no-mamba" "--no-cl" "--no-iochunter")
NAMES=("main" "no_mamba" "no_cl" "no_iochunter")
FAILED=()

for i in "${!FLAGS[@]}"; do
  name="${NAMES[$i]}"; flag="${FLAGS[$i]}"
  echo
  echo "${B}==================== [$((i + 1))/4] $name ====================${N}"
  if [ -z "$flag" ]; then
    python -m src.train "$@"
  else
    python -m src.train "$flag" "$@"
  fi
  if [ $? -ne 0 ]; then
    echo "${R}[$name] TRAINING FAILED${N}"; FAILED+=("$name")
  fi
done

if [ ${#FAILED[@]} -gt 0 ]; then
  echo; echo "${R}Failed configurations: ${FAILED[*]} - stopping without evaluation.${N}"; exit 1
fi

echo
echo "${B}==================== Stage 6 - evaluation ====================${N}"
python -m src.evaluate all --robustness --validity || exit 1

echo
echo "${G}Done. Results: outputs/metrics_all.csv, robustness_*.csv, validity_*.csv, tsne_*.png${N}"
echo "Per-epoch history: outputs/history_*.csv (the train_cl column shows whether CL is learning)"
