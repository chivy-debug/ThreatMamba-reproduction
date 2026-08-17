"""Stage 5: train the classifier on D2 (Table V): Adam lr 1e-4, 100 epochs, valid 0.1.
Loss = BCE + lambda * CL (Eq. 18-20).

THREE BUGS FIXED (the first run of the `main` model failed acceptance; see the README):
  1. cl_lambda = 1.0 let the contrastive term SWAMP the BCE term. With K=4 negatives,
     InfoNCE starts around ln(K+1) = 1.61 while BCE starts around 0.35, so more than 80%
     of the gradient came from the contrastive objective and the model never learned to
     classify. New default: 0.1 (the first point of the {0.1, 0.5, 1.0} grid).
  2. Early stopping on validation MACRO-F1 is far too noisy: 20 classes over ~250
     validation samples, many classes with only 2-4 samples. `main` hit a lucky "record"
     at epoch 2 and stopped at epoch 7 while another ablation ran 41 epochs, which makes
     the ablation table meaningless. Selection now uses balanced accuracy (macro recall):
     it still weights rare classes like macro-F1, but is smoother because there is no
     precision term collapsing to 0 when a class is never predicted. min_epochs was added.
  3. The contrastive term had no warm-up. The first cl_warmup_epochs epochs now run BCE
     only, so the model learns to classify before the contrastive objective kicks in.

CLI:
  python -m src.train                       # main model (tag: main)
  python -m src.train --no-mamba            # ablation w/o MAMBA
  python -m src.train --no-cl               # ablation w/o contrastive learning
  python -m src.train --no-iochunter        # ablation w/o IOCHunter (drops co_occurs/hunting edges)
  python -m src.train --cl-lambda 0.5 --suffix lam05     # lambda grid search
Checkpoint: outputs/model_{tag}.pt | History: outputs/history_{tag}.csv
"""
import argparse
import csv
import math
import random

import torch

from .cskg_builder import CSKG_DIR, REL
from .contrastive import neg_splice, pos_view
from .losses import bce_loss, info_nce
from .model import ThreatMambaModel, collate
from .utils import (ATTCK_DIR, OUTPUTS, PROCESSED, load_config, load_json, read_jsonl,
                    set_seed, torch_load)

HUNT_RELS = {REL["co_occurs"], REL["rapiddns_history"], REL["otx_general"],
             REL["vt_object"], REL["vt_behavior"]}

# Criteria for picking the best checkpoint. Lower is better for 'loss', higher for the rest.
SELECT_METRICS = ("balanced_acc", "micro", "macro", "loss")


def tag_of(args) -> str:
    base = ("no_mamba" if args.no_mamba else
            "no_cl" if args.no_cl else
            "no_iochunter" if args.no_iochunter else "main")
    return f"{base}_{args.suffix}" if args.suffix else base


def load_graphs(split: str, limit=None) -> list[dict]:
    docs = read_jsonl(PROCESSED / "docs.jsonl")
    docs = [d for d in docs if d["split"] == split]
    if limit:
        docs = docs[:limit]
    out = []
    for d in docs:
        fp = CSKG_DIR / f"{d['doc_id']}.pt"
        if fp.exists():
            out.append(torch_load(fp))
    print(f"{split}: {len(out)}/{len(docs)} CSKGs")
    return out


def macro_f1(y_true: list[int], y_pred: list[int], n_cls: int) -> float:
    """Kept for backwards compatibility with older code and tests."""
    from sklearn.metrics import f1_score
    return f1_score(y_true, y_pred, labels=list(range(n_cls)), average="macro", zero_division=0)


def cls_metrics(y_true: list[int], y_pred: list[int], n_cls: int) -> dict:
    """micro-F1 (= accuracy), macro-F1, balanced accuracy (= macro recall), and the number
    of classes actually predicted (an early warning for a degenerate, near-constant model)."""
    from sklearn.metrics import f1_score, recall_score
    L = list(range(n_cls))
    present = sorted(set(y_true))
    return {
        "micro": float(f1_score(y_true, y_pred, labels=L, average="micro", zero_division=0)),
        "macro": float(f1_score(y_true, y_pred, labels=L, average="macro", zero_division=0)),
        "balanced_acc": float(recall_score(y_true, y_pred, labels=present,
                                           average="macro", zero_division=0)),
        "n_pred_cls": float(len(set(y_pred))),
    }


def run_epoch(model, graphs, labels, groups, cfg, args, device, opt=None, rng=None,
              tac_order=None, lam=0.0) -> dict:
    """Run one epoch. Returns a dict with loss/bce/cl and micro/macro/balanced_acc/n_pred_cls.
    `lam` is supplied by the CALLER so that warm-up and --no-cl follow the same code path."""
    train = opt is not None
    model.train() if train else model.eval()
    bs = int(cfg["train"].get("batch_size", 16))
    tau = float(args.tau if args.tau is not None else cfg["train"]["cl_temperature"])
    K = int(cfg["train"]["cl_pairs_K"])
    noise = float(cfg["train"].get("cl_noise_std", 0.05))
    edrop = float(cfg["train"].get("cl_edge_drop", 0.1))
    drop_rels = HUNT_RELS if args.no_iochunter else None
    order = list(range(len(graphs)))
    if train:
        rng.shuffle(order)
    s_bce, s_cl, n, n_cl = 0.0, 0.0, 0, 0
    preds, gts = [], []
    C = len(groups)
    for i in range(0, len(order), bs):
        ids = order[i:i + bs]
        gs = [graphs[j] for j in ids]
        y = torch.tensor([labels[j] for j in ids], device=device)
        y1h = torch.zeros(len(ids), C, device=device)
        y1h[torch.arange(len(ids)), y] = 1
        batch = collate(gs, device, drop_rels)
        l_cl = None
        with torch.set_grad_enabled(train):
            logits, z = model(batch)
            l_bce = bce_loss(logits, y1h)
            loss = l_bce
            if train and lam > 0 and len(ids) >= 2:
                pos = [pos_view(g, noise, edrop, rng) for g in gs]
                _, z_pos = model(collate(pos, device, drop_rels))
                negs = []
                for a, ga in enumerate(gs):
                    cands = [b for b in range(len(gs)) if labels[ids[b]] != labels[ids[a]]] or \
                            [b for b in range(len(gs)) if b != a]
                    row = [neg_splice(ga, gs[rng.choice(cands)], tac_order, rng=rng) for _ in range(K)]
                    negs.append(row)
                flat = [g for row in negs for g in row]
                _, z_neg = model(collate(flat, device, drop_rels))
                z_neg = z_neg.view(len(gs), K, -1)
                l_cl = info_nce(z, z_pos, z_neg, tau)
                loss = l_bce + lam * l_cl
            if train:
                opt.zero_grad(); loss.backward(); opt.step()
        s_bce += float(l_bce) * len(ids); n += len(ids)
        if l_cl is not None:
            s_cl += float(l_cl) * len(ids); n_cl += len(ids)
        preds += logits.argmax(-1).tolist(); gts += y.tolist()
    met = cls_metrics(gts, preds, C)
    met["bce"] = s_bce / max(1, n)
    met["cl"] = (s_cl / n_cl) if n_cl else 0.0
    met["loss"] = met["bce"] + lam * met["cl"]
    return met


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None, help="debug: cap the number of graphs")
    ap.add_argument("--no-mamba", action="store_true")
    ap.add_argument("--no-cl", action="store_true")
    ap.add_argument("--no-iochunter", action="store_true")
    ap.add_argument("--cl-lambda", type=float, default=None, help="override train.cl_lambda")
    ap.add_argument("--tau", type=float, default=None, help="override train.cl_temperature")
    ap.add_argument("--patience", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--select-metric", choices=SELECT_METRICS, default=None,
                    help="checkpoint selection criterion (defaults to the config value)")
    ap.add_argument("--suffix", default=None, help="tag suffix, used for grid search")
    args = ap.parse_args()
    cfg = load_config(args.config)
    seed = int(args.seed if args.seed is not None else cfg["data"].get("seed", 42))
    set_seed(seed)
    rng = random.Random(seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    tag = tag_of(args)

    groups = load_json(PROCESSED / "groups.json")
    g2i = {g: i for i, g in enumerate(groups)}
    graphs = load_graphs("train", args.limit)
    labels = [g2i[g["group"]] for g in graphs]
    idx = list(range(len(graphs)))
    rng.shuffle(idx)
    n_val = max(1, int(len(idx) * float(cfg["train"]["valid_split"])))
    val_idx, tr_idx = idx[:n_val], idx[n_val:]
    tr_graphs = [graphs[i] for i in tr_idx]; tr_labels = [labels[i] for i in tr_idx]
    va_graphs = [graphs[i] for i in val_idx]; va_labels = [labels[i] for i in val_idx]

    attck = load_json(ATTCK_DIR / "attck_v14.json") if (ATTCK_DIR / "attck_v14.json").exists() else None
    tac_order = [t["shortname"] for t in attck["tactic_order"]] if attck else []

    m = cfg["model"]
    model = ThreatMambaModel(len(groups), d=int(m["gat_dim"]), heads=int(m["gat_heads"]),
                             gat_layers=int(m["gat_layers"]), ssm_layers=int(m["ssm_layers"]),
                             dropout=float(m["dropout"]), ssm_mode=m.get("ssm_fallback", "auto"),
                             no_mamba=args.no_mamba).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=float(cfg["train"]["lr"]))

    # --- contrastive configuration + stopping criterion ---
    lam_cfg = 0.0 if args.no_cl else float(
        args.cl_lambda if args.cl_lambda is not None else cfg["train"]["cl_lambda"])
    warm = 0 if args.no_cl else int(cfg["train"].get("cl_warmup_epochs", 0))
    K = int(cfg["train"]["cl_pairs_K"])
    sel = args.select_metric or str(cfg["train"].get("select_metric", "balanced_acc"))
    if sel not in SELECT_METRICS:
        raise SystemExit(f"select_metric must be one of {SELECT_METRICS}, got {sel!r}")
    lower_better = (sel == "loss")
    epochs = int(args.epochs or cfg["train"]["epochs"])
    patience = int(args.patience if args.patience is not None
                   else cfg["train"]["early_stopping_patience"])
    min_epochs = int(cfg["train"].get("min_epochs", 0))

    print(f"[{tag}] train {len(tr_graphs)} / val {len(va_graphs)} | {len(groups)} groups | device={device}")
    print(f"[{tag}] lambda={lam_cfg:g} tau={args.tau or cfg['train']['cl_temperature']} K={K} "
          f"warmup={warm} | select on val {sel} | patience={patience} min_epochs={min_epochs}")
    if lam_cfg > 0:
        print(f"[{tag}] note: the chance level of InfoNCE is ln(K+1) = {math.log(K + 1):.4f}. "
              f"If the 'cl' column sits at that value, contrastive learning is not learning.")

    best = math.inf if lower_better else -math.inf
    best_ep, best_val, wait = 0, None, 0
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    hist_fp = OUTPUTS / f"history_{tag}.csv"
    cols = ["epoch", "lambda", "train_loss", "train_bce", "train_cl", "train_micro",
            "train_macro", "train_bal_acc", "val_micro", "val_macro", "val_bal_acc",
            "val_n_pred_cls", "is_best"]
    with open(hist_fp, "w", newline="") as hf:
        wr = csv.writer(hf); wr.writerow(cols)
        for ep in range(1, epochs + 1):
            lam = 0.0 if ep <= warm else lam_cfg          # warm-up: BCE only
            tr = run_epoch(model, tr_graphs, tr_labels, groups, cfg, args, device,
                           opt=opt, rng=rng, tac_order=tac_order, lam=lam)
            with torch.no_grad():
                va = run_epoch(model, va_graphs, va_labels, groups, cfg, args, device)

            # validation runs with lam=0, so va["loss"] is exactly the BCE on the val split
            score = va[sel]
            better = (score < best) if lower_better else (score > best)
            if better:
                best, best_ep, best_val, wait = score, ep, dict(va), 0
                torch.save({"state_dict": model.state_dict(), "groups": groups,
                            "flags": {"no_mamba": args.no_mamba, "no_cl": args.no_cl,
                                      "no_iochunter": args.no_iochunter},
                            # record which SSM implementation was used ("mamba" or "simple")
                            # so loading never rebuilds a mismatched architecture on a machine
                            # that does or does not have mamba-ssm available
                            "ssm_impl": (model.ssm.mode if model.ssm is not None else None),
                            "model_cfg": dict(m),
                            # the hyper-parameters actually used, so results stay traceable
                            "train_cfg": {"cl_lambda": lam_cfg, "cl_warmup_epochs": warm,
                                          "cl_temperature": float(args.tau or cfg["train"]["cl_temperature"]),
                                          "cl_pairs_K": K, "select_metric": sel,
                                          "seed": seed, "best_epoch": ep},
                            "val_metrics": dict(va)}, OUTPUTS / f"model_{tag}.pt")
            else:
                wait += 1

            wr.writerow([ep, f"{lam:g}", f"{tr['loss']:.4f}", f"{tr['bce']:.4f}", f"{tr['cl']:.4f}",
                         f"{tr['micro']:.4f}", f"{tr['macro']:.4f}", f"{tr['balanced_acc']:.4f}",
                         f"{va['micro']:.4f}", f"{va['macro']:.4f}", f"{va['balanced_acc']:.4f}",
                         int(va["n_pred_cls"]), int(better)]); hf.flush()
            print(f"[{tag}] ep {ep:3d} | lam {lam:<4g} | loss {tr['loss']:.4f} "
                  f"(bce {tr['bce']:.4f} cl {tr['cl']:.4f}) | tr bal {tr['balanced_acc']:.4f} | "
                  f"val micro {va['micro']:.4f} macro {va['macro']:.4f} bal {va['balanced_acc']:.4f} "
                  f"| pred {int(va['n_pred_cls'])}/{len(groups)}{'  *' if better else ''}")

            if wait >= patience and ep >= min_epochs:
                print(f"[{tag}] early stopping at epoch {ep} (patience {patience}, "
                      f"best epoch {best_ep})")
                break

    v = best_val or {}
    print(f"\n[{tag}] BEST epoch {best_ep} by val {sel} = {best:.4f}")
    print(f"[{tag}] val micro {v.get('micro', 0):.4f} | macro {v.get('macro', 0):.4f} | "
          f"bal_acc {v.get('balanced_acc', 0):.4f} | predicted {int(v.get('n_pred_cls', 0))}/{len(groups)} classes")
    if v.get("n_pred_cls", 0) <= max(2, len(groups) // 4):
        print(f"[{tag}] WARNING: the model only predicts {int(v['n_pred_cls'])}/{len(groups)} classes - "
              f"close to degenerate. Lower cl_lambda (currently {lam_cfg:g}) or train longer.")
    print(f"[{tag}] checkpoint: outputs/model_{tag}.pt | history: outputs/history_{tag}.csv")


if __name__ == "__main__":
    main()
