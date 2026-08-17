#!/usr/bin/env bash
# Stage 1 - download the source data: MuscleFish/ThreatMAMBA + MITRE ATT&CK v14 (STIX).
# Usage:  bash scripts/download_data.sh
set -euo pipefail
cd "$(dirname "$0")/.."

echo "-- [1/2] MuscleFish/ThreatMAMBA -> data/raw/MuscleFish"
if [ ! -f data/raw/MuscleFish/CTI2Attacker.csv ]; then
  rm -rf /tmp/tm_mf data/raw/MuscleFish
  git clone --depth 1 https://github.com/MuscleFish/ThreatMAMBA /tmp/tm_mf
  mkdir -p data/raw/MuscleFish data/reference
  cp /tmp/tm_mf/*.csv /tmp/tm_mf/*.xlsx data/raw/MuscleFish/ 2>/dev/null || true
  cp /tmp/tm_mf/pie-node-ttp-state-graph.temp.html app/pie_node_graph.html
  cp /tmp/tm_mf/TTP_Group_Contribution.csv data/reference/ttp_group_contribution_authors.csv
  rm -rf /tmp/tm_mf
fi
ls -la data/raw/MuscleFish/

echo "-- [2/2] ATT&CK v14 (enterprise-attack-14.1.json) -> data/attck/"
if [ ! -f data/attck/enterprise-attack-14.1.json ]; then
  rm -rf /tmp/tm_asd
  mkdir -p /tmp/tm_asd && cd /tmp/tm_asd
  git init -q
  git remote add origin https://github.com/mitre-attack/attack-stix-data
  git config core.sparseCheckout true
  git sparse-checkout set --no-cone '/enterprise-attack/enterprise-attack-14.1.json'
  git fetch -q --depth 1 --filter=blob:none origin master
  git checkout -q FETCH_HEAD
  cd - > /dev/null
  cp /tmp/tm_asd/enterprise-attack/enterprise-attack-14.1.json data/attck/
  rm -rf /tmp/tm_asd
fi
du -h data/attck/enterprise-attack-14.1.json

echo
echo "Done. Next:  python -m src.data_prep all"
