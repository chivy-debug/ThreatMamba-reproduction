"""Stage 6 explainability:
- Node contribution scores (state/TTP/IOC) in the spirit of Eq. 21-23: gradient x input on
  the node features with respect to the predicted class logit. This is a simplification of
  the paper's formulation and is documented as such.
- TTP x group heatmap in the style of Fig. 5.
- Jaccard + F1 against the MITRE ATT&CK Groups v14 pages (Eq. 28-29), producing a table in
  the style of Table XI (the Qi'anxin column is omitted).

CLI:
  python -m src.explain doc --doc d00042 [--tag main]
  python -m src.explain group-profile [--tag main]     # heatmap + CSV
  python -m src.explain attck-match [--tag main]       # Table XI
"""
import argparse
import csv
import json
from collections import defaultdict

import numpy as np
import torch

from .cskg_builder import CSKG_DIR, NODE_TYPES
from .evaluate import load_model
from .model import collate
from .train import HUNT_RELS, load_graphs
from .utils import ATTCK_DIR, OUTPUTS, PROCESSED, load_config, load_json, torch_load


def node_contributions(model, g: dict, device, drop_rels=None) -> tuple[int, np.ndarray]:
    """Returns (predicted class, per-node contribution scores >= 0, normalised to sum to 1)."""
    batch = collate([g], device, drop_rels)
    batch["x"].requires_grad_(True)
    logits, _ = model(batch)
    pred = int(logits.argmax(-1)[0])
    model.zero_grad(set_to_none=True)
    logits[0, pred].backward()
    grad = batch["x"].grad
    score = (grad * batch["x"]).abs().sum(-1).detach().cpu().numpy()
    total = score.sum() or 1.0
    return pred, score / total


def cmd_doc(args, cfg, device):
    model, groups, fl = load_model(args.tag, cfg, device)
    drop = HUNT_RELS if fl["no_iochunter"] else None
    fp = CSKG_DIR / f"{args.doc}.pt"
    g = torch_load(fp)
    pred, score = node_contributions(model, g, device, drop)
    print(f"doc {args.doc} (ground truth: {g['group']}) -> predicted: {groups[pred]}")
    rank = np.argsort(-score)
    rows = []
    for i in rank[:25]:
        t = NODE_TYPES[g["node_types"][i]]
        rows.append({"node": g["node_values"][i], "type": t, "score": round(float(score[i]), 5)})
        print(f"  {score[i]:.4f}  {t:<13} {g['node_values'][i][:60]}")
    out = OUTPUTS / f"explain_{args.doc}.json"
    json.dump({"doc_id": args.doc, "true": g["group"], "pred": groups[pred], "top_nodes": rows},
              open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"-> {out}")


def _group_tech_scores(model, graphs, groups, device, drop):
    """Aggregate Technique-node contribution scores by (true group, technique)."""
    acc: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    tech_t = NODE_TYPES.index("Technique")
    for g in graphs:
        _pred, score = node_contributions(model, g, device, drop)
        for i, t in enumerate(g["node_types"]):
            if t == tech_t:
                acc[g["group"]][g["node_values"][i]].append(float(score[i]))
    return {grp: {tech: float(np.mean(v)) for tech, v in d.items()} for grp, d in acc.items()}


def cmd_group_profile(args, cfg, device):
    model, groups, fl = load_model(args.tag, cfg, device)
    drop = HUNT_RELS if fl["no_iochunter"] else None
    graphs = load_graphs("test")
    gts = _group_tech_scores(model, graphs, groups, device, drop)
    techs = sorted({t for d in gts.values() for t in d})
    if not techs:
        raise SystemExit("No Technique nodes in the test CSKGs "
                         "(run Stage 3, then rebuild the CSKGs)")
    grps = sorted(gts)
    M = np.zeros((len(grps), len(techs)))
    for i, grp in enumerate(grps):
        for j, t in enumerate(techs):
            M[i, j] = gts[grp].get(t, 0.0)
    with open(OUTPUTS / "fig5_group_ttp_contribution.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["group"] + techs)
        for i, grp in enumerate(grps):
            w.writerow([grp] + [round(x, 5) for x in M[i]])
    top = np.argsort(-M.sum(0))[:40]
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.figure(figsize=(14, 7))
    plt.imshow(M[:, top], aspect="auto", cmap="YlOrRd")
    plt.yticks(range(len(grps)), grps, fontsize=7)
    plt.xticks(range(len(top)), [techs[j] for j in top], rotation=90, fontsize=6)
    plt.colorbar(label="mean contribution score")
    plt.title("TTP contribution x group (Fig. 5 style)")
    plt.tight_layout(); plt.savefig(OUTPUTS / "fig5_heatmap.png", dpi=180); plt.close()
    print("-> outputs/fig5_heatmap.png + fig5_group_ttp_contribution.csv")


def _attck_group_map(groups, attck):
    """Map MuscleFish group names to ATT&CK intrusion-sets via name/alias, case-insensitively."""
    by_alias = {}
    for name, info in attck["groups"].items():
        for al in [name] + info.get("aliases", []):
            by_alias.setdefault(al.lower(), name)
    # Manual mapping for dataset group names that do not match any ATT&CK alias.
    # Verified against ATT&CK v14: Nitro / SEA / Scarab / TwoForOne have NO corresponding
    # intrusion-set, so they are left unmapped and reported as such.
    manual = {"hidden cobra": "Lazarus Group", "waterbug": "Turla",
              "energetic bear": "Dragonfly", "quedagh": "Sandworm Team",
              "tick": "BRONZE BUTLER", "magecart": "FIN6"}
    out = {}
    for g in groups:
        key = g.lower()
        out[g] = by_alias.get(key) or by_alias.get(manual.get(key, "").lower())
    return out


def cmd_attck_match(args, cfg, device):
    model, groups, fl = load_model(args.tag, cfg, device)
    drop = HUNT_RELS if fl["no_iochunter"] else None
    attck = load_json(ATTCK_DIR / "attck_v14.json")
    graphs = load_graphs("test")
    gts = _group_tech_scores(model, graphs, groups, device, drop)
    if not gts:
        raise SystemExit("No Technique nodes in the test CSKGs - run Stage 3 "
                         "(ttp_extract train) and rebuild the CSKGs before matching "
                         "against ATT&CK")
    top_n = int(cfg["eval"].get("explain_top_ttps", 20))
    gmap = _attck_group_map(sorted(gts), attck)
    rows = []
    for grp in sorted(gts):
        ours = {t.split(".")[0] for t, _s in
                sorted(gts[grp].items(), key=lambda kv: -kv[1])[:top_n]}
        aname = gmap.get(grp)
        if not aname:
            rows.append({"group": grp, "attck_group": "(unmapped)", "n_ours": len(ours),
                         "n_attck": 0, "jaccard": "", "f1": ""})
            continue
        theirs = {t.split(".")[0] for t in attck["groups"][aname]["techniques"]}
        inter = ours & theirs
        jac = len(inter) / max(1, len(ours | theirs))
        prec = len(inter) / max(1, len(ours)); rec = len(inter) / max(1, len(theirs))
        f1 = 2 * prec * rec / max(1e-9, prec + rec)
        rows.append({"group": grp, "attck_group": aname, "n_ours": len(ours),
                     "n_attck": len(theirs), "jaccard": round(jac, 4), "f1": round(f1, 4)})
        print(f"{grp:<15} ~ {aname:<22} J={jac:.3f} F1={f1:.3f} ({len(inter)}/{len(ours)}/{len(theirs)})")
    with open(OUTPUTS / "tableXI_attck_match.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    vals = [r["jaccard"] for r in rows if r["jaccard"] != ""]
    if vals:
        print(f"mean Jaccard: {np.mean(vals):.4f} -> outputs/tableXI_attck_match.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["doc", "group-profile", "attck-match"])
    ap.add_argument("--doc", default=None)
    ap.add_argument("--tag", default="main")
    ap.add_argument("--config", default=None)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    {"doc": cmd_doc, "group-profile": cmd_group_profile, "attck-match": cmd_attck_match}[args.cmd](args, cfg, device)


if __name__ == "__main__":
    main()
