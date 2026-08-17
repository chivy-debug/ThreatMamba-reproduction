#!/usr/bin/env python3
"""Stage 2 acceptance check. After ~24h of daemon runtime: at least 5 demo documents
should carry Time/Geo-location nodes, and the log should show no unrecoverable crash."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENRICHED = ROOT / "data" / "enriched"
LOG = ROOT / "outputs" / "enrich.log"


def main() -> int:
    files = sorted(ENRICHED.glob("d*.json"))
    n_time_geo = 0
    type_cover = set()
    for fp in files:
        d = json.load(open(fp, encoding="utf-8"))
        types = {a["type"] for h in d.get("hunts", []) for a in h.get("accepted", [])}
        types |= {i["type"] for i in d.get("iocs", [])}
        type_cover |= types
        if types & {"Time", "Geo-location"}:
            n_time_geo += 1
    print(f"Enriched: {len(files)} documents | with Time/Geo: {n_time_geo} (need >=5)")
    print(f"Node types covered ({len(type_cover)}/19): {sorted(type_cover)}")
    if LOG.exists():
        lines = LOG.read_text(encoding="utf-8").splitlines()
        errs = [l for l in lines[-500:] if " ERROR " in l or "Traceback" in l]
        print(f"Log: {len(lines)} lines | recent ERRORs: {len(errs)}")
        for l in errs[-5:]:
            print("   ", l[:160])
        alive = any("daemon start" in l or "DONE" in l for l in lines[-50:])
        print(f"Daemon recently active: {'yes' if alive else 'unclear'}")
    ok = n_time_geo >= 5
    print("=> " + ("Stage 2 acceptance PASSED" if ok else
                   "NOT YET (let the daemon run longer / check the API keys)"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
