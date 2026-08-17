#!/usr/bin/env python3
"""Pick demo documents that the model reliably gets right, for live presentations.

Why this script exists: the model reaches roughly 54% top-1 accuracy, so opening a random
document during a demo is close to a coin flip - especially for the rare groups
(Darkhotel has 32 documents, Energetic Bear 39, while FIN7 has 576). This script scores
every document in the test split and selects N of them that are:

  - correctly attributed at top-1,
  - predicted with a wide margin (top-1 probability well above top-2),
  - backed by a rich CSKG (more nodes and states make the graph and robustness pages
    look meaningful),
  - each from a DIFFERENT group, for variety,
  - still correct at the 60% timeline cut-off, so the robustness page does not fall apart.

Usage:
  python scripts/pick_demo_docs.py                  # pick 5, overwrite data/demo_subset.txt
  python scripts/pick_demo_docs.py -n 8 --dry-run   # preview only, write nothing
  python scripts/pick_demo_docs.py --tag main --min-keep 0.4

Outputs: data/demo_subset.txt (the doc_id list) and outputs/demo_candidates.csv (full table).
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.cskg_builder import truncate_graph                      # noqa: E402
from src.evaluate import load_model, predict                     # noqa: E402
from src.train import HUNT_RELS, load_graphs                     # noqa: E402
from src.utils import OUTPUTS, PROCESSED, load_config, load_json, read_jsonl  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", "--num", type=int, default=5, help="how many documents to pick")
    ap.add_argument("--tag", default="main")
    ap.add_argument("--config", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--min-keep", type=float, default=0.6,
                    help="document must still be correct at this timeline cut-off "
                         "(0 disables the check)")
    ap.add_argument("--min-nodes", type=int, default=25,
                    help="minimum CSKG node count, so the graph renders well")
    ap.add_argument("--per-group", type=int, default=1, help="max documents per group")
    ap.add_argument("--dry-run", action="store_true", help="print only, write no files")
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if not (OUTPUTS / f"model_{args.tag}.pt").exists():
        raise SystemExit(f"outputs/model_{args.tag}.pt not found - run Stage 5 first.")

    groups = load_json(PROCESSED / "groups.json")
    g2i = {g: i for i, g in enumerate(groups)}
    graphs = load_graphs("test")
    if not graphs:
        raise SystemExit("No CSKGs for the test split - run Stage 4 first.")
    y = np.array([g2i[g["group"]] for g in graphs])

    model, _g, fl = load_model(args.tag, cfg, device)
    drop = HUNT_RELS if fl["no_iochunter"] else None

    print(f"Scoring {len(graphs)} test documents with model '{args.tag}'...")
    probs, _rep, _z = predict(model, graphs, device, drop)

    # Re-measure at the min_keep cut-off to drop documents that are only correct once the
    # full timeline is available.
    keep_ok = np.ones(len(graphs), dtype=bool)
    if args.min_keep > 0:
        sub = [truncate_graph(g, args.min_keep) for g in graphs]
        p_cut, _r, _z2 = predict(model, sub, device, drop)
        keep_ok = p_cut.argmax(-1) == y
        print(f"Still correct at {int(args.min_keep * 100)}% of the timeline: "
              f"{int(keep_ok.sum())}/{len(graphs)}")

    docs = {d["doc_id"]: d for d in read_jsonl(PROCESSED / "docs.jsonl")}
    order = np.argsort(-probs, axis=-1)
    keep_col = f"correct_at_{int(args.min_keep * 100)}pct"
    rows = []
    for i, g in enumerate(graphs):
        p = probs[i]
        top1, top2 = order[i, 0], order[i, 1]
        doc = docs.get(g["doc_id"], {})
        rows.append({
            "doc_id": g["doc_id"], "group": g["group"],
            "pred": groups[top1], "correct": int(top1 == y[i]),
            "prob_top1": round(float(p[top1]), 4),
            "margin": round(float(p[top1] - p[top2]), 4),
            "n_nodes": len(g["node_types"]), "n_states": len(g["states"]),
            "n_sentences": len(doc.get("text", "").split(".")),
            keep_col: int(keep_ok[i]),
        })

    rows.sort(key=lambda r: -r["margin"])
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    with open(OUTPUTS / "demo_candidates.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    pool = [r for r in rows if r["correct"] and r["n_nodes"] >= args.min_nodes
            and (args.min_keep <= 0 or r[keep_col])]
    if not pool:
        print(f"\nNo document satisfies every condition. Relax them: lower --min-nodes "
              f"(currently {args.min_nodes}) or --min-keep (currently {args.min_keep}).")
        pool = [r for r in rows if r["correct"]]

    picked, used = [], {}
    for r in pool:
        if used.get(r["group"], 0) >= args.per_group:
            continue
        picked.append(r); used[r["group"]] = used.get(r["group"], 0) + 1
        if len(picked) >= args.num:
            break
    # If too few groups qualify, relax the one-document-per-group constraint.
    for r in pool:
        if len(picked) >= args.num:
            break
        if r not in picked:
            picked.append(r)

    print(f"\n{'doc_id':<10} {'ground truth':<16} {'predicted':<16} {'top-1':>7} {'margin':>8} "
          f"{'nodes':>6} {'states':>7}")
    print("-" * 78)
    for r in picked:
        print(f"{r['doc_id']:<10} {r['group']:<16} {r['pred']:<16} {r['prob_top1']:>7.3f} "
              f"{r['margin']:>8.3f} {r['n_nodes']:>6} {r['n_states']:>7}")

    if args.dry_run:
        print("\n--dry-run: no files written.")
        return

    fp = PROCESSED.parent / "demo_subset.txt"
    if fp.exists():
        fp.with_suffix(".txt.bak").write_text(fp.read_text(encoding="utf-8"), encoding="utf-8")
        print(f"\n(previous version backed up to {fp.with_suffix('.txt.bak')})")
    header = ("# Demo documents - selected by scripts/pick_demo_docs.py\n"
              f"# model={args.tag}  criteria: top-1 correct, still correct at "
              f"{int(args.min_keep * 100)}% of the timeline, >= {args.min_nodes} nodes\n")
    fp.write_text(header + "\n".join(r["doc_id"] for r in picked) + "\n", encoding="utf-8")
    print(f"-> {fp} ({len(picked)} documents)")
    print(f"-> outputs/demo_candidates.csv (full table of {len(rows)} documents "
          f"if you want to pick others)")


if __name__ == "__main__":
    main()
