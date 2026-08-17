#!/usr/bin/env bash
# ==========================================================================
# Grid search over lambda for the contrastive loss (Eq. 18). The paper does not state a
# value, so a small grid search is run and the selected value recorded.
#
# Usage:  bash scripts/grid_cl.sh                 # {0.1, 0.5, 1.0}
#         LAMS="0.05 0.1 0.2" bash scripts/grid_cl.sh
#         bash scripts/grid_cl.sh --epochs 40     # extra args are forwarded to src.train
#
# Outputs: outputs/model_main_lamXX.pt + outputs/grid_cl_summary.csv
# Once a lambda is chosen, write it into configs/default.yaml (train.cl_lambda) and run
#   bash scripts/train_all.sh
# to retrain all four ablations with that value.
# ==========================================================================
set -uo pipefail
cd "$(dirname "$0")/.."

if [ -t 1 ]; then G=$'\e[32m'; B=$'\e[1m'; N=$'\e[0m'; else G=""; B=""; N=""; fi
[ -f .venv/bin/activate ] && . .venv/bin/activate

LAMS="${LAMS:-0.1 0.5 1.0}"
TAGS=()

for lam in $LAMS; do
  sfx="lam${lam//./}"
  echo; echo "${B}==================== lambda = $lam  (tag: main_$sfx) ====================${N}"
  python -m src.train --cl-lambda "$lam" --suffix "$sfx" "$@" || exit 1
  TAGS+=("main_$sfx")
done

echo; echo "${B}==================== Evaluating on the test split ====================${N}"
for t in "${TAGS[@]}"; do
  python -m src.evaluate "$t" || exit 1
done

python - "$LAMS" <<'PY'
import csv, sys
from pathlib import Path
out = Path("outputs")
lams = sys.argv[1].split()
rows = []
for lam in lams:
    tag = "main_lam" + lam.replace(".", "")
    fp = out / f"metrics_{tag}.csv"
    if not fp.exists():
        continue
    r = next(csv.DictReader(open(fp)))
    r = {"cl_lambda": lam, **r}
    rows.append(r)
if not rows:
    raise SystemExit("No results found")
with open(out / "grid_cl_summary.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print()
print(f"{'lambda':>8} {'macro-F1':>9} {'micro-F1':>9} {'top1':>7} {'top3':>7} {'#classes pred':>14}")
for r in rows:
    print(f"{r['cl_lambda']:>8} {float(r['f1_macro']):>9.4f} {float(r['f1_micro']):>9.4f} "
          f"{float(r['top1_micro']):>7.4f} {float(r['top3_micro']):>7.4f} "
          f"{int(float(r.get('n_pred_cls', 0))):>14}")
best = max(rows, key=lambda r: float(r["f1_macro"]))
print(f"\nBest by macro-F1: lambda = {best['cl_lambda']} "
      f"(macro-F1 {float(best['f1_macro']):.4f})")
print("-> Write this value into configs/default.yaml (train.cl_lambda), then run:")
print("   bash scripts/train_all.sh")
print("-> outputs/grid_cl_summary.csv")
PY
echo; echo "${G}Grid search complete.${N}"
