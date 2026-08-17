#!/usr/bin/env bash
# Canonical Stage 1-7 sequence on the target machine (after Stage 0 has passed).
# Every stage can be run on its own; this script is the ordering + acceptance reminder.
set -e
cd "$(dirname "$0")/.."
PY=.venv/bin/python; [ -x "$PY" ] || PY=python3

echo "== STAGE 1 - data =="
bash scripts/download_data.sh
"$PY" -m src.data_prep all         # ACCEPTANCE: statistics table + ATT&CK >=600 techniques / 14 tactics

echo "== STAGE 2 - IOCHunter daemon (runs in the background throughout) =="
bash scripts/run_enrichment_daemon.sh    # ACCEPTANCE after 24h: python scripts/check_enrichment.py

echo "== STAGE 3 - TTP extraction =="
"$PY" -m src.ttp_extract encode
"$PY" -m src.ttp_extract train
"$PY" -m src.ttp_extract eval             # ACCEPTANCE: tactics micro F1 >= 0.70

echo "== STAGE 4 - CSKG (MUST run after Stage 3 so TTP nodes exist) =="
"$PY" -m src.cskg_builder build --split all --mode train
"$PY" -m src.cskg_builder stats           # ACCEPTANCE: ~53 nodes / 147 edges, same order of magnitude
"$PY" -m src.cskg_builder build --split all --mode enriched
"$PY" -m src.cskg_builder render --doc "$(head -1 data/demo_subset.txt)"

echo "== STAGE 5 - train the classifier (run overnight) =="
# train_all.sh trains all four configurations with the SAME config, seed and stopping
# criterion, then evaluates them, so the ablation table is actually meaningful.
bash scripts/train_all.sh                 # ACCEPTANCE: macro-F1 > 0.35, Top-3 > 0.60

echo "== STAGE 6 - explainability =="
"$PY" -m src.explain group-profile        # Fig. 5 heatmap
"$PY" -m src.explain attck-match          # Table XI

echo "== STAGE 7 - demo UI =="
"$PY" tests/test_ui_smoke.py              # ACCEPTANCE: all 4 pages error-free + timing check
echo "Open the UI with:  streamlit run app/streamlit_app.py"
echo "Done. Results are in outputs/"
