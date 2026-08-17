#!/usr/bin/env python3
"""Stage 7 UI smoke test: actually runs the Streamlit script for all four pages via
AppTest and reports any page that raises.

Run:  python tests/test_ui_smoke.py

This script deliberately does NOT set THREATMAMBA_SSM / THREATMAMBA_FAKE_ENCODER: the
Stage 7 acceptance check must run the real model and the real encoder, otherwise the
timing numbers (<60s per document) are meaningless. Forcing THREATMAMBA_SSM=simple would
also break loading a checkpoint trained with mamba.

Set them by hand only when testing on a machine WITHOUT a GPU, and only if a `simple`
checkpoint exists:
  THREATMAMBA_FAKE_ENCODER=1 THREATMAMBA_SSM=simple python tests/test_ui_smoke.py

Note: a page missing data from an earlier stage shows a warning and calls st.stop().
That is CORRECT behaviour and is not counted as a failure.
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

APP = ROOT / "app" / "streamlit_app.py"


def main() -> int:
    from streamlit.testing.v1 import AppTest

    if not (ROOT / "outputs" / "model_main.pt").exists():
        print("[SKIP] outputs/model_main.pt not found - run Stage 5 first")
        return 2

    at = AppTest.from_file(str(APP), default_timeout=180).run()
    if at.exception:
        print(f"[FAIL] default page: {at.exception[0].message}")
        return 1
    nav = at.sidebar.radio[0]          # the page selector radio lives in the sidebar
    pages = list(nav.options)
    print(f"Found {len(pages)} pages: {pages}")

    fails = 0
    for p in pages:
        at.sidebar.radio[0].set_value(p).run()
        if at.exception:
            print(f"[FAIL] {p}: {at.exception[0].message}")
            fails += 1
            continue
        n_warn = len(at.warning)
        widgets = (len(at.dataframe) + len(at.selectbox) + len(at.slider)
                   + len(at.metric) + len(at.button) + len(at.radio))
        note = "  (stopped early: earlier-stage data missing - expected)" if n_warn else ""
        print(f"[PASS] {p}: {widgets} widgets, {n_warn} warnings{note}")

    print("-" * 50)
    fails += timing_check()
    print("=" * 50)
    print(f"{len(pages) - fails}/{len(pages)} pages ran without errors" if fails <= len(pages)
          else "failures detected")
    return 1 if fails else 0


def timing_check() -> int:
    """Stage 7 acceptance: a fresh document must take < 60s (without live enrichment),
    and each slider position < 2s thanks to caching."""
    import time

    from src.inference import InferenceService, load_demo_doc
    from src.utils import PROCESSED

    subset_fp = PROCESSED.parent / "demo_subset.txt"
    doc_id = (subset_fp.read_text(encoding="utf-8").split()[0] if subset_fp.exists()
              else "d00000")
    try:
        d = load_demo_doc(doc_id)
    except SystemExit:
        print("[SKIP] timing: Stage 1 data not available")
        return 0

    svc = InferenceService()
    t0 = time.time()
    svc.analyze(d["text"], d["doc_id"], d["group"])
    t_full = time.time() - t0
    worst = 0.0
    for k in (0.2, 0.4, 0.6, 0.8, 1.0):
        t1 = time.time()
        svc.analyze(d["text"], d["doc_id"], d["group"], keep_frac=k)
        worst = max(worst, time.time() - t1)
    ok1, ok2 = t_full < 60, worst < 2.0
    print(f"[{'PASS' if ok1 else 'FAIL'}] end-to-end, one document: {t_full:.2f}s (need < 60s)")
    print(f"[{'PASS' if ok2 else 'FAIL'}] slowest slider position: {worst:.2f}s (need < 2s)")
    return 0 if (ok1 and ok2) else 1


if __name__ == "__main__":
    sys.exit(main())
