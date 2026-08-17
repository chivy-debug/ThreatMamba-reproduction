"""Stage 6 evaluation: micro/macro F1, Top-1/3/5 (micro and macro), robustness masking
(Table VIII + Eq. 27), D_intra/D_inter/D_separ (Eq. 24-26, Table X), t-SNE (Fig. 4).

CLI:
  python -m src.evaluate main                      # headline metrics -> outputs/metrics_main.csv
  python -m src.evaluate main --robustness         # 5 timeline cut-offs + fit m = At + B
  python -m src.evaluate main --validity           # D_intra/inter/separ + t-SNE
  python -m src.evaluate all                       # main + 3 ablations (summary table)
"""
import argparse
import csv

import numpy as np
import torch

from .cskg_builder import truncate_graph
from .model import ThreatMambaModel, collate
from .train import HUNT_RELS, load_graphs
from .utils import OUTPUTS, PROCESSED, load_config, load_json, torch_load

TAGS = ["main", "no_mamba", "no_cl", "no_iochunter"]


def load_model(tag: str, cfg, device):
    ck = torch_load(OUTPUTS / f"model_{tag}.pt", map_location=device)
    m = ck["model_cfg"]; fl = ck["flags"]
    impl = ck.get("ssm_impl") or m.get("ssm_fallback", "auto")   # match the training-time impl
    model = ThreatMambaModel(len(ck["groups"]), d=int(m["gat_dim"]), heads=int(m["gat_heads"]),
                             gat_layers=int(m["gat_layers"]), ssm_layers=int(m["ssm_layers"]),
                             dropout=float(m["dropout"]), ssm_mode=impl,
                             no_mamba=fl["no_mamba"]).to(device)
    try:
        model.load_state_dict(ck["state_dict"])
    except RuntimeError as e:
        got = model.ssm.mode if model.ssm is not None else "no SSM"
        raise SystemExit(
            f"Cannot load outputs/model_{tag}.pt: SSM architecture mismatch.\n"
            f"  Checkpoint was trained with : {impl}\n"
            f"  Model was just built with   : {got}\n"
            f"Most common cause: the THREATMAMBA_SSM environment variable is forcing a "
            f"different mode.\n"
            f"  Check:  echo $THREATMAMBA_SSM        (should be empty)\n"
            f"  Clear:  unset THREATMAMBA_SSM\n"
            f"If this machine has no GPU/mamba-ssm but the checkpoint was trained with "
            f"mamba, it CANNOT be loaded - retrain with THREATMAMBA_SSM=simple.\n"
            f"(original error: {str(e)[:200]}...)") from None
    model.eval()
    return model, ck["groups"], fl


def predict(model, graphs, device, drop_rels=None, bs=32):
    """Returns (probs, V_G, z).
    V_G = the graph representation fed to the classification MLP (Eq. 15-17); this is what
          Eq. 24-26 and the t-SNE plot are computed on.
    z   = the contrastive projection head output, used ONLY for InfoNCE."""
    probs, reps, zs = [], [], []
    with torch.no_grad():
        for i in range(0, len(graphs), bs):
            batch = collate(graphs[i:i + bs], device, drop_rels)
            logits, z, rep = model(batch, return_rep=True)
            probs.append(torch.sigmoid(logits).cpu())
            reps.append(rep.cpu()); zs.append(z.cpu())
    return torch.cat(probs).numpy(), torch.cat(reps).numpy(), torch.cat(zs).numpy()


def validity(vec: np.ndarray, y: np.ndarray, n_cls: int) -> tuple[float, float, float, list[int]]:
    """Eq. 24-26 over the vectors in `vec`.
    Returns (D_intra, D_inter, D_separ, classes present)."""
    vc = vec - vec.mean(0, keepdims=True)
    present = [c for c in range(n_cls) if (y == c).any()]
    cent = np.stack([vc[y == c].mean(0) for c in present])
    d_intra = float(np.mean([np.linalg.norm(vc[y == c] - cent[i], axis=1).mean()
                             for i, c in enumerate(present)]))
    dd = [np.linalg.norm(cent[i] - cent[j])
          for i in range(len(cent)) for j in range(i + 1, len(cent))]
    d_inter = float(np.mean(dd)) if dd else 0.0
    return d_intra, d_inter, d_inter / max(d_intra, 1e-9), present


def topk_metrics(probs: np.ndarray, y: np.ndarray, n_cls: int, ks=(1, 3, 5)) -> dict:
    from sklearn.metrics import f1_score
    pred = probs.argmax(-1)
    out = {"f1_micro": f1_score(y, pred, average="micro", zero_division=0),
           "f1_macro": f1_score(y, pred, labels=list(range(n_cls)), average="macro", zero_division=0)}
    order = np.argsort(-probs, axis=-1)
    for k in ks:
        hit = np.any(order[:, :k] == y[:, None], axis=1)
        out[f"top{k}_micro"] = float(hit.mean())
        per_cls = [hit[y == c].mean() for c in range(n_cls) if (y == c).any()]
        out[f"top{k}_macro"] = float(np.mean(per_cls))
    # How many classes the model ACTUALLY predicts. If this is far below n_cls the model is
    # close to constant, and every other metric (including a "great robustness" curve) is
    # untrustworthy.
    out["n_pred_cls"] = float(len(np.unique(pred)))
    return out


def eval_tag(tag: str, graphs, y, groups, cfg, device):
    model, _g, fl = load_model(tag, cfg, device)
    drop = HUNT_RELS if fl["no_iochunter"] else None
    probs, rep, z = predict(model, graphs, device, drop)
    met = topk_metrics(probs, y, len(groups))
    return model, probs, rep, z, met, drop


def cmd_eval(args, cfg, device):
    graphs = load_graphs("test")
    groups = load_json(PROCESSED / "groups.json")
    g2i = {g: i for i, g in enumerate(groups)}
    y = np.array([g2i[g["group"]] for g in graphs])
    tags = TAGS if args.tag == "all" else [args.tag]
    rows = []
    for tag in tags:
        fp = OUTPUTS / f"model_{tag}.pt"
        if not fp.exists():
            print(f"[{tag}] SKIP (no checkpoint)")
            continue
        model, probs, rep, z, met, drop = eval_tag(tag, graphs, y, groups, cfg, device)
        rows.append({"model": tag, **{k: round(v, 4) for k, v in met.items()}})
        print(f"[{tag}] " + "  ".join(f"{k}={v:.4f}" for k, v in met.items()))
        n_pred = int(met["n_pred_cls"])
        degenerate = n_pred <= max(2, len(groups) // 4)
        if degenerate:
            print(f"   !! WARNING: only {n_pred}/{len(groups)} classes predicted - the model is "
                  f"close to degenerate. Every number below (robustness included) is "
                  f"unreliable. Lower train.cl_lambda and retrain.")

        if args.robustness:
            ks = [float(k) for k in cfg["eval"]["robustness_keep"]]
            rrows = []
            for kf in ks:
                sub = [truncate_graph(g, kf) for g in graphs]
                p, _r, _z = predict(model, sub, device, drop)
                m = topk_metrics(p, y, len(groups))
                rrows.append({"keep_frac": kf, **{k: round(v, 4) for k, v in m.items()}})
                print(f"   keep {int(kf * 100):3d}%: top1 {m['top1_micro']:.4f}  top3 {m['top3_micro']:.4f}")
            xs = np.array([r["keep_frac"] for r in rrows]); ys = np.array([r["top1_micro"] for r in rrows])
            # Table VIII of the paper reports m as a PERCENTAGE. Fitting on the % scale keeps
            # A/B directly comparable with the paper (ThreatMAMBA: A=37.54 B=30.32, micro Top-1).
            A, B = np.polyfit(xs, ys * 100.0, 1)
            print(f"   fit Eq.27 (% scale): m = {A:.2f}*t + {B:.2f}")
            print(f"      large A = exploits new information well as the timeline fills in")
            print(f"      large B = already accurate in the early, data-poor stage")
            print(f"      paper ThreatMAMBA: A=37.54 B=30.32 | GAT: A=28.63 B=25.20")
            # INTERPRETATION TRAP: a degenerate model predicts almost a constant, so truncating
            # the timeline changes nothing. The curve is perfectly flat and A ~ 0, which LOOKS
            # like "extremely robust". This is FAKE robustness and must be flagged before the
            # numbers end up in a Table VIII write-up.
            t3 = np.array([r["top3_micro"] for r in rrows])
            if degenerate or (np.ptp(ys) < 1e-6 and np.ptp(t3) < 1e-6):
                print(f"   !! FAKE ROBUSTNESS: top1/top3 are identical at all {len(rrows)} "
                      f"timeline cut-offs. A={A:.2f} is small NOT because the model is robust "
                      f"but because it ignores its input. Do not report this table.")
            with open(OUTPUTS / f"robustness_{tag}.csv", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rrows[0]) + ["fit_A", "fit_B"])
                w.writeheader()
                for r in rrows:
                    w.writerow({**r, "fit_A": round(A, 4), "fit_B": round(B, 4)})

        if args.validity:
            # Eq. 24-26 are computed on V_G (the CSKG vector fed to the classification MLP),
            # exactly as in the paper. The *_z columns are for reference only: measured on the
            # contrastive projection head they diverge sharply, because InfoNCE L2-normalises,
            # keeping only ANGULAR information and discarding magnitude.
            d_intra, d_inter, d_separ, lab_present = validity(rep, y, len(groups))
            zi, ze, zs_ = validity(z, y, len(groups))[:3]
            print(f"   [V_G]      D_intra={d_intra:.4f}  D_inter={d_inter:.4f}  D_separ={d_separ:.4f}"
                  f"   (paper ThreatMAMBA: 10.862 / 33.436 / 3.078)")
            print(f"   [z / CL]   D_intra={zi:.4f}  D_inter={ze:.4f}  D_separ={zs_:.4f}"
                  f"   (reference only, do not report)")
            with open(OUTPUTS / f"validity_{tag}.csv", "w", newline="") as f:
                csv.writer(f).writerows(
                    [["D_intra", "D_inter", "D_separ", "D_intra_z", "D_inter_z", "D_separ_z"],
                     [round(d_intra, 4), round(d_inter, 4), round(d_separ, 4),
                      round(zi, 4), round(ze, 4), round(zs_, 4)]])
            try:
                from sklearn.manifold import TSNE
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt
                perp = min(30, max(2, len(rep) // 4))
                t2 = TSNE(n_components=2, random_state=42, perplexity=perp).fit_transform(rep)
                plt.figure(figsize=(9, 7))
                for c in lab_present:
                    pts = t2[y == c]
                    plt.scatter(pts[:, 0], pts[:, 1], s=12, label=groups[c])
                plt.legend(fontsize=6, ncol=2, markerscale=1.5)
                plt.title(f"t-SNE of V_G - {tag} (Fig. 4)"); plt.tight_layout()
                plt.savefig(OUTPUTS / f"tsne_{tag}.png", dpi=180); plt.close()
                print(f"   t-SNE -> outputs/tsne_{tag}.png")
            except Exception as e:  # noqa: BLE001
                print(f"   t-SNE failed: {e}")

    if rows:
        with open(OUTPUTS / ("metrics_all.csv" if args.tag == "all" else f"metrics_{args.tag}.csv"),
                  "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader(); w.writerows(rows)
        print(f"-> outputs/metrics_*.csv")
        if args.tag == "all":
            main_r = next((r for r in rows if r["model"] == "main"), None)
            if main_r:
                ok = main_r["f1_macro"] > 0.35 and main_r["top3_micro"] > 0.60
                print(f"STAGE 5 ACCEPTANCE: macro-F1 {main_r['f1_macro']} (>0.35?) "
                      f"top3 {main_r['top3_micro']} (>0.60?) => {'PASS' if ok else 'FAIL'}")
                # An ablation beating the main model is a sign of MISCONFIGURATION, not a
                # scientific finding. If removing a component improves results substantially,
                # that component is switched on incorrectly (e.g. cl_lambda far too high)
                # rather than being useless.
                better = [r for r in rows if r["model"] != "main"
                          and r["f1_macro"] > main_r["f1_macro"] + 0.02]
                for r in better:
                    print(f"   !! ABLATION '{r['model']}' BEATS the main model "
                          f"(macro-F1 {r['f1_macro']} vs {main_r['f1_macro']}). "
                          f"Check the configuration of the removed component before "
                          f"drawing conclusions.")
                if any(r["model"] == "no_cl" for r in better):
                    print("   -> Suspect train.cl_lambda is too high. Run scripts/grid_cl.sh "
                          "to re-measure lambda over {0.1, 0.5, 1.0}.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tag", help="main | no_mamba | no_cl | no_iochunter | all | "
                                "any tag name (e.g. main_lam05 from a grid search)")
    ap.add_argument("--robustness", action="store_true")
    ap.add_argument("--validity", action="store_true")
    ap.add_argument("--config", default=None)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    if args.tag != "all" and not (OUTPUTS / f"model_{args.tag}.pt").exists():
        have = sorted(p.stem.replace("model_", "") for p in OUTPUTS.glob("model_*.pt"))
        raise SystemExit(f"outputs/model_{args.tag}.pt not found. Available tags: {have or '(none)'}")
    cfg = load_config(args.config)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    cmd_eval(args, cfg, device)


if __name__ == "__main__":
    main()
