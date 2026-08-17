#!/usr/bin/env python3
"""Selectively prune the Stage 2 daemon state: keep the good results, drop only the
broken entries.

WHY THIS EXISTS:
  The first daemon run (before .env was loaded correctly) left behind:
   - ~155 documents marked "done" despite yielding no nodes at all
   - ~2,400 IOCs cached with empty results, so they would never be looked up again
  Deleting _state.json outright would also throw away the documents that did succeed.
  This script removes only the EMPTY entries and leaves everything else untouched.

USAGE
  python scripts/prune_enrich_state.py            # dry run: show what would be dropped
  python scripts/prune_enrich_state.py --apply    # actually prune (a backup is written)
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENRICHED = ROOT / "data" / "enriched"
STATE = ENRICHED / "_state.json"


def doc_has_nodes(doc_id: str) -> bool:
    fp = ENRICHED / f"{doc_id}.json"
    if not fp.exists():
        return False
    try:
        d = json.load(open(fp, encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return False
    return any(h.get("accepted") for h in d.get("hunts", []))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--keep-vt-counter", action="store_true",
                    help="keep the VirusTotal quota counter (default: reset it, because the "
                         "earlier run miscounted 480 phantom calls)")
    args = ap.parse_args()

    if not STATE.exists():
        print(f"{STATE} not found - nothing to prune.")
        return 0
    st = json.load(open(STATE, encoding="utf-8"))

    done = list(st.get("done_docs", []))
    keep_done = [d for d in done if doc_has_nodes(d)]
    drop_done = [d for d in done if d not in set(keep_done)]

    cache = dict(st.get("ioc_cache", {}))
    keep_cache = {k: v for k, v in cache.items()
                  if any(h.get("accepted") for h in (v or []))}
    n_drop_cache = len(cache) - len(keep_cache)

    empty_files = [p for p in sorted(ENRICHED.glob("d*.json"))
                   if not doc_has_nodes(p.stem)]

    uct = st.get("uct", {})
    cov = sum((uct.get("coverage") or {}).values())

    print("=" * 62)
    print(f"done_docs : {len(done)} -> keeping {len(keep_done)} (dropping {len(drop_done)} empty docs)")
    print(f"ioc_cache : {len(cache)} -> keeping {len(keep_cache)} (dropping {n_drop_cache} empty entries)")
    print(f"empty JSON files to delete: {len(empty_files)}")
    print(f"UCT counters: current total coverage = {cov}"
          f"{'  (=0 -> will be reset so UCT relearns from scratch)' if cov == 0 else ''}")
    print(f"vt_calls  : {st.get('vt_calls')}")
    if not args.apply:
        print("\n(re-run with --apply to actually prune)")
        return 0

    shutil.copy2(STATE, STATE.with_suffix(".json.bak"))
    st["done_docs"] = keep_done
    st["ioc_cache"] = keep_cache
    st.pop("failed_docs", None)
    if cov == 0:
        st["uct"] = {}                       # all-zero coverage is meaningless; relearn
    if not args.keep_vt_counter:
        st["vt_calls"] = {}                  # the old run counted 480 calls that never left
    json.dump(st, open(STATE, "w", encoding="utf-8"))
    for p in empty_files:
        p.unlink()
    print(f"\nPruned. Backup: {STATE.with_suffix('.json.bak').name}")
    print(f"{len(keep_done)} genuinely enriched documents and {len(keep_cache)} cached IOCs remain.")
    print("Continue with:  bash scripts/run_enrichment_daemon.sh")
    return 0


if __name__ == "__main__":
    sys.exit(main())
