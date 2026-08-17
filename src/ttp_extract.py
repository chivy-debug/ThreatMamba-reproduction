"""Stage 3 TTP extraction module: SecureBERT -> V_E projection -> SSM -> per-label
Gaussian attention -> multi-label sigmoid (Eq. 2-5). The Gaussian prototype uses a
DIAGONAL Sigma, which is a deliberate simplification of the paper.

CLI:
  python -m src.ttp_extract encode              # precompute D1 embeddings (once, cached)
  python -m src.ttp_extract train               # train on D1
  python -m src.ttp_extract eval                # Stage 3 acceptance check
  python -m src.ttp_extract extract --text "..."
  python -m src.ttp_extract stats               # D1 label statistics (diagnostics)

FOUR FIXES OVER THE FIRST IMPLEMENTATION (each of them alone forced F1 = 0.0000):
 1) Early stopping on macro average precision (threshold-free) instead of F1@0.5.
    F1 was always 0, so "never better than the record" stopped training at epoch 6/100.
 2) Decision thresholds are TUNED ON THE VALIDATION SET rather than fixed at 0.5.
    D1 is extremely sparse (3.06 positive labels out of 347 = 0.88%); the most frequent
    label, TA0005, appears in only 38.3% of samples, so nothing ever crossed 0.5 and every
    prediction was 0.
 3) BCE uses pos_weight so the positive class is not drowned out by the negatives.
 4) The TTP module gets its own d_model (default 256). Projecting 768 -> 48 and then
    discriminating 347 labels is far too tight; 48 is the Stage 5 classifier width from
    Table V, not this module's.

API used from Stage 4 onwards:
  TTPService.extract(text) -> {"ttps": [(id, prob)], "seq": [(sent_idx, id)]}
"""
import argparse
import json
import random
from collections import Counter

import numpy as np
import torch
import torch.nn as nn

from .encoder import encode_doc_cached, get_encoder
from .ssm import SSMStack
from .utils import OUTPUTS, PROCESSED, load_config, read_jsonl, set_seed, torch_load

D1_EMB = PROCESSED / "d1_emb.pt"
CKPT = OUTPUTS / "ttp_extractor.pt"


# ------------------------------------------------------------------ model
class TTPExtractor(nn.Module):
    def __init__(self, n_labels: int, d_in: int = 768, d: int = 256,
                 ssm_layers: int = 3, ssm_mode: str = "auto", dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(d_in, d), nn.LayerNorm(d), nn.GELU(),
                                  nn.Dropout(dropout))
        self.ssm = SSMStack(d, ssm_layers, ssm_mode, dropout)
        self.mu = nn.Parameter(torch.randn(n_labels, d) * 0.1)      # Gaussian prototype
        self.logvar = nn.Parameter(torch.zeros(n_labels, d))        # diagonal Sigma
        self.w = nn.Parameter(torch.randn(n_labels, d) * 0.1)
        self.b = nn.Parameter(torch.zeros(n_labels))

    def attention(self, h: torch.Tensor, mask: torch.Tensor):
        """h (B,S,d) -> attn (B,S,L): weights ~ N(h; mu_l, diag(var_l)), softmaxed over S."""
        inv = torch.exp(-self.logvar.clamp(-6, 6))                   # (L,d)
        A = (h ** 2) @ inv.t()                                       # (B,S,L)
        Bt = h @ (self.mu * inv).t()                                 # (B,S,L)
        C = (self.mu ** 2 * inv).sum(-1) + self.logvar.sum(-1)       # (L,)
        logw = -0.5 * (A - 2 * Bt + C) / max(1, h.shape[-1]) ** 0.5
        logw = logw.masked_fill(~mask.bool().unsqueeze(-1), -1e4)
        return torch.softmax(logw, dim=1)

    def forward(self, emb: torch.Tensor, mask: torch.Tensor):
        """emb (B,S,768), mask (B,S) -> logits (B,L), attn (B,S,L)."""
        h = self.proj(emb) * mask.unsqueeze(-1)
        h = self.ssm(h, mask)
        attn = self.attention(h, mask)
        r = torch.einsum("bsl,bsd->bld", attn, h)                    # (B,L,d)
        logits = (r * self.w).sum(-1) + self.b
        return logits, attn


# ------------------------------------------------------------------ data
def load_label_space(min_pos: int = 5):
    rows = read_jsonl(PROCESSED / "d1.jsonl")
    space = json.load(open(PROCESSED / "d1_label_space.json", encoding="utf-8"))
    pos = Counter(l for r in rows for l in r["labels"])
    tactics = space["tactics"]
    techniques = [t for t in space["techniques"] if pos[t] >= min_pos]
    return rows, tactics + techniques, tactics, techniques


def _split_idx(n: int, seed: int = 42):
    idx = list(range(n))
    random.Random(seed).shuffle(idx)
    n_te, n_va = int(n * 0.1), int(n * 0.1)
    return idx[n_te + n_va:], idx[n_te:n_te + n_va], idx[:n_te]      # train, val, test


def _label_matrix(rows, labels):
    lab2i = {l: i for i, l in enumerate(labels)}
    Y = torch.zeros(len(rows), len(labels))
    for i, r in enumerate(rows):
        for l in r["labels"]:
            if l in lab2i:
                Y[i, lab2i[l]] = 1
    return Y


def _batches(order, embs, Y, bs, device, shuffle=False, seed=0):
    order = list(order)
    if shuffle:
        random.Random(seed).shuffle(order)
    for i in range(0, len(order), bs):
        ids = order[i:i + bs]
        seqs = [embs[j].float() for j in ids]
        S = max((s.shape[0] for s in seqs), default=1) or 1
        emb = torch.zeros(len(ids), S, 768)
        mask = torch.zeros(len(ids), S)
        for k, s in enumerate(seqs):
            if s.shape[0]:
                emb[k, :s.shape[0]] = s
                mask[k, :s.shape[0]] = 1
        yield emb.to(device), mask.to(device), Y[ids].to(device)


def _predict(model, order, embs, Y, bs, device):
    model.eval()
    probs, ys = [], []
    with torch.no_grad():
        for emb, mask, y in _batches(order, embs, Y, bs, device):
            logits, _ = model(emb, mask)
            probs.append(torch.sigmoid(logits).float().cpu().numpy())
            ys.append(y.cpu().numpy())
    return np.concatenate(probs), np.concatenate(ys)


# ------------------------------------------------------------------ thresholds + metrics
def macro_ap(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Macro average precision: threshold-free, used as the early-stopping criterion."""
    from sklearn.metrics import average_precision_score
    cols = [j for j in range(y_true.shape[1]) if y_true[:, j].sum() > 0]
    if not cols:
        return 0.0
    return float(np.mean([average_precision_score(y_true[:, j], y_prob[:, j]) for j in cols]))


def _f1_at(y_true, y_prob, thr) -> float:
    from sklearn.metrics import f1_score
    return float(f1_score(y_true, (y_prob >= thr).astype(int), average="micro", zero_division=0))


def tune_thresholds(y_true: np.ndarray, y_prob: np.ndarray, n_tactics: int,
                    per_label_min_pos: int = 20) -> dict:
    """Tune decision thresholds on the VALIDATION set:
       - one shared threshold for tactics and one for techniques (maximising micro F1)
       - per-label refinement for labels with enough positives (>= per_label_min_pos)
    Returns {"tactics": float, "techniques": float, "per_label": {index: float}}."""
    grid = np.arange(0.02, 0.90, 0.01)
    out = {}
    for name, sl in (("tactics", slice(0, n_tactics)), ("techniques", slice(n_tactics, None))):
        yt, yp = y_true[:, sl], y_prob[:, sl]
        if yt.size == 0 or yt.sum() == 0:
            out[name] = 0.5
            continue
        scores = [_f1_at(yt, yp, t) for t in grid]
        out[name] = float(grid[int(np.argmax(scores))])
    per_label = {}
    from sklearn.metrics import f1_score
    for j in range(y_true.shape[1]):
        if y_true[:, j].sum() < per_label_min_pos:
            continue
        base = out["tactics"] if j < n_tactics else out["techniques"]
        best_t, best_f = base, f1_score(y_true[:, j], (y_prob[:, j] >= base).astype(int),
                                        zero_division=0)
        for t in grid:
            f = f1_score(y_true[:, j], (y_prob[:, j] >= t).astype(int), zero_division=0)
            if f > best_f:
                best_f, best_t = f, float(t)
        per_label[j] = best_t
    out["per_label"] = per_label
    return out


def apply_thresholds(y_prob: np.ndarray, thr: dict, n_tactics: int) -> np.ndarray:
    t = np.full(y_prob.shape[1], float(thr.get("techniques", 0.5)))
    t[:n_tactics] = float(thr.get("tactics", 0.5))
    for j, v in (thr.get("per_label") or {}).items():
        t[int(j)] = float(v)
    return (y_prob >= t[None, :]).astype(int)


def _pos_weight(Y_train: torch.Tensor, cap: float) -> torch.Tensor:
    pos = Y_train.sum(0).clamp(min=1.0)
    neg = Y_train.shape[0] - pos
    return (neg / pos).clamp(max=cap)


# ------------------------------------------------------------------ commands
def cmd_stats(args):
    cfg = load_config(args.config)
    rows, labels, tactics, techs = load_label_space(cfg["ttp"]["min_label_pos"])
    Y = _label_matrix(rows, labels)
    n_lab = Y.sum(1)
    print(f"D1: {len(rows)} samples | {len(labels)} labels ({len(tactics)} tactics + {len(techs)} techniques)")
    print(f"positive labels/sample: mean {n_lab.mean():.2f} | overall positive rate {Y.mean():.4%}")
    print(f"tactics only: positive rate {Y[:, :len(tactics)].mean():.2%}")
    freq = Y.mean(0)
    top = torch.argsort(freq, descending=True)[:8]
    print("most frequent labels:", [(labels[i], f"{freq[i]:.1%}") for i in top.tolist()])
    print(f"=> With a fixed threshold of 0.5 NO label would ever fire ({freq.max():.1%} is the highest)")


def cmd_encode(args):
    cfg = load_config(args.config)
    from .utils import sent_split
    rows, _, _, _ = load_label_space(cfg["ttp"]["min_label_pos"])
    enc = get_encoder()
    max_s = int(cfg["ttp"]["max_sents"])
    embs = []
    for i, r in enumerate(rows):
        sents = sent_split(r["text"])[:max_s] or [r["text"][:512]]
        embs.append(encode_doc_cached(f"d1_{i:05d}", sents, enc).half())
        if (i + 1) % 2000 == 0:
            print(f"  encoded {i + 1}/{len(rows)}")
    torch.save(embs, D1_EMB)
    print(f"OK: {len(embs)} samples -> {D1_EMB}")


def cmd_train(args):
    cfg = load_config(args.config)
    tcfg, trcfg = cfg["ttp"], cfg["train"]
    set_seed(int(cfg["data"].get("seed", 42)))
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    rows, labels, tactics, techs = load_label_space(tcfg["min_label_pos"])
    if not D1_EMB.exists():
        raise SystemExit("d1_emb.pt not found - run: python -m src.ttp_extract encode")
    embs = torch_load(D1_EMB, weights_only=True)
    Y = _label_matrix(rows, labels)
    tr, va, te = _split_idx(len(rows))
    n_tac = len(tactics)
    print(f"D1: {len(rows)} samples | {len(labels)} labels ({n_tac} tactics + {len(techs)} techniques) "
          f"| train/val/test = {len(tr)}/{len(va)}/{len(te)} | device={device}")
    print(f"    positive rate: {Y.mean():.4%} (tactics {Y[:, :n_tac].mean():.2%})")

    model = TTPExtractor(len(labels), d=int(tcfg["d_model"]), ssm_layers=int(tcfg["ssm_layers"]),
                         ssm_mode=cfg["model"].get("ssm_fallback", "auto"),
                         dropout=float(cfg["model"]["dropout"])).to(device)
    lr = float(tcfg.get("lr", trcfg["lr"]))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    pw = _pos_weight(Y[tr], float(tcfg.get("pos_weight_cap", 50))).to(device)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pw)
    print(f"    lr={lr} | pos_weight: mean {pw.mean():.1f}, max {pw.max():.1f}")

    bs = int(tcfg["batch_size"])
    patience = int(trcfg["early_stopping_patience"])
    epochs = int(args.epochs or trcfg["epochs"])
    best_ap, wait, best_state = -1.0, 0, None
    for ep in range(1, epochs + 1):
        model.train()
        tot = n = 0
        for emb, mask, y in _batches(tr, embs, Y, bs, device, shuffle=True, seed=ep):
            logits, _ = model(emb, mask)
            loss = lossf(logits, y)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot += float(loss) * len(y); n += len(y)
        yp, yt = _predict(model, va, embs, Y, bs, device)
        ap = macro_ap(yt, yp)
        ap_tac = macro_ap(yt[:, :n_tac], yp[:, :n_tac])
        print(f"epoch {ep:3d} | loss {tot / max(1, n):.4f} | val AP macro {ap:.4f} "
              f"(tactics {ap_tac:.4f})")
        if ap > best_ap + 1e-5:
            best_ap, wait = ap, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                print(f"early stopping (patience {patience})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    yp, yt = _predict(model, va, embs, Y, bs, device)
    thr = tune_thresholds(yt, yp, n_tac)
    pred = apply_thresholds(yp, thr, n_tac)
    from sklearn.metrics import f1_score
    print(f"thresholds tuned on val: tactics={thr['tactics']:.2f} techniques={thr['techniques']:.2f} "
          f"| per-label refinement for {len(thr['per_label'])} labels")
    print(f"val F1 micro (tactics) = "
          f"{f1_score(yt[:, :n_tac], pred[:, :n_tac], average='micro', zero_division=0):.4f}")

    OUTPUTS.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "labels": labels,
                "tactics": tactics, "techniques": techs,
                "ssm_impl": model.ssm.mode, "thresholds": thr,
                "cfg": {"d": int(tcfg["d_model"]), "ssm_layers": int(tcfg["ssm_layers"])}}, CKPT)
    print(f"best val AP macro = {best_ap:.4f} | checkpoint: {CKPT}")


def cmd_eval(args):
    cfg = load_config(args.config)
    tcfg = cfg["ttp"]
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    rows, labels, _tac, _te = load_label_space(tcfg["min_label_pos"])
    embs = torch_load(D1_EMB, weights_only=True)
    ck = torch_load(CKPT, map_location=device)
    model = TTPExtractor(len(ck["labels"]), d=ck["cfg"]["d"], ssm_layers=ck["cfg"]["ssm_layers"],
                         ssm_mode=ck.get("ssm_impl") or cfg["model"].get("ssm_fallback", "auto")).to(device)
    model.load_state_dict(ck["state_dict"])
    Y = _label_matrix(rows, ck["labels"])
    n_tac = len(ck["tactics"])
    _tr, _va, te = _split_idx(len(rows))
    yp, yt = _predict(model, te, embs, Y, int(tcfg["batch_size"]), device)
    thr = ck.get("thresholds") or {"tactics": 0.5, "techniques": 0.5, "per_label": {}}
    pred = apply_thresholds(yp, thr, n_tac)

    from sklearn.metrics import f1_score
    res = {
        "tactics_f1_micro": f1_score(yt[:, :n_tac], pred[:, :n_tac], average="micro", zero_division=0),
        "tactics_f1_macro": f1_score(yt[:, :n_tac], pred[:, :n_tac], average="macro", zero_division=0),
        "tactics_AP_macro": macro_ap(yt[:, :n_tac], yp[:, :n_tac]),
        "tech_f1_micro": f1_score(yt[:, n_tac:], pred[:, n_tac:], average="micro", zero_division=0),
        "tech_f1_macro": f1_score(yt[:, n_tac:], pred[:, n_tac:], average="macro", zero_division=0),
        "tech_AP_macro": macro_ap(yt[:, n_tac:], yp[:, n_tac:]),
    }
    print("=" * 56)
    print("STAGE 3 ACCEPTANCE - D1 test split")
    print(f"  thresholds in use: tactics={thr['tactics']:.2f} techniques={thr['techniques']:.2f}")
    for k, v in res.items():
        print(f"  {k:<18} {v:.4f}")
    print(f"  mean positive predictions/sample: {pred.sum(1).mean():.2f} (ground truth: {yt.sum(1).mean():.2f})")
    f1 = res["tactics_f1_micro"]
    goal = float(cfg["ttp"].get("accept_tactics_f1", 0.70))
    print(f"=> tactics micro F1 = {f1:.4f} | acceptance threshold {goal}: "
          f"{'PASS' if f1 >= goal else 'FAIL'}")
    if f1 < goal:
        print("   If all four fixes are in place and the score is still low, revisit the")
        print("   expectation: D1 labels are SENTENCE-level (~1 sentence / 90 chars per sample).")


# ------------------------------------------------------------------ service
class TTPService:
    def __init__(self, ckpt=CKPT, device=None, config=None):
        cfg = load_config(config)
        self.cfg = cfg["ttp"]
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        ck = torch_load(ckpt, map_location=self.device)
        self.labels = ck["labels"]
        self.n_tac = len(ck["tactics"])
        self.thr = ck.get("thresholds") or {"tactics": 0.5, "techniques": 0.5, "per_label": {}}
        self.model = TTPExtractor(len(self.labels), d=ck["cfg"]["d"],
                                  ssm_layers=ck["cfg"]["ssm_layers"],
                                  ssm_mode=ck.get("ssm_impl")
                                  or cfg["model"].get("ssm_fallback", "auto")).to(self.device)
        self.model.load_state_dict(ck["state_dict"]); self.model.eval()
        self.encoder = get_encoder(device=str(self.device))
        t = np.full(len(self.labels), float(self.thr.get("techniques", 0.5)))
        t[:self.n_tac] = float(self.thr.get("tactics", 0.5))
        for j, v in (self.thr.get("per_label") or {}).items():
            t[int(j)] = float(v)
        self._thr_vec = t

    def extract_from_emb(self, emb: torch.Tensor) -> dict:
        """DOCUMENT MODE: treat the whole sentence sequence as ONE sample (S,768)
        producing a single prediction vector.

        WARNING: the model is trained on D1, where 97% of samples are a SINGLE sentence.
        Feeding a 60-128 sentence document here dilutes the Gaussian attention and in
        practice returns 0 TTPs. Kept for comparison; extract_per_sentence is the default.
        """
        S = emb.shape[0]
        if S == 0:
            return {"ttps": [], "seq": []}
        mask = torch.ones(1, S, device=self.device)
        with torch.no_grad():
            logits, attn = self.model(emb.unsqueeze(0).to(self.device), mask)
            probs = torch.sigmoid(logits)[0].float().cpu().numpy()
            attn = attn[0].float().cpu().numpy()
        hit = np.nonzero(probs >= self._thr_vec)[0]
        ttps = [(self.labels[j], float(probs[j])) for j in hit]
        seq = [(int(attn[:, j].argmax()), self.labels[j]) for j in hit]
        seq.sort(key=lambda t: (t[0], t[1]))
        return {"ttps": sorted(ttps, key=lambda t: -t[1]), "seq": seq}

    def extract_per_sentence(self, sents: list[str], emb: torch.Tensor | None = None) -> dict:
        """SENTENCE MODE (default): matches the distribution the model was trained on.

        Each sentence is an independent sample (B, 1, 768). Advantages over document mode:
          - matches the training distribution, so the thresholds tuned on validation hold
          - gives an EXACT sent_idx (which sentence the TTP came from) instead of guessing
            it from the attention argmax, which makes the CSKG time axis (Eq. 6) correct
        emb: when called from cskg_builder, the cached embedding is passed through so the
        document is not encoded twice.
        """
        if not sents:
            return {"ttps": [], "seq": []}
        bs = int(self.cfg.get("batch_size", 32))
        max_per_sent = int(self.cfg.get("max_ttps_per_sent", 5))
        max_per_doc = int(self.cfg.get("max_ttps_per_doc", 40))
        best: dict[str, tuple[float, int]] = {}      # label -> (highest prob, first sent_idx)
        with torch.no_grad():
            for i in range(0, len(sents), bs):
                chunk_emb = (emb[i:i + bs] if emb is not None
                             else self.encoder.encode(sents[i:i + bs]))
                if chunk_emb.shape[0] == 0:
                    continue
                x = chunk_emb.float().unsqueeze(1).to(self.device)     # (n,1,768)
                mask = torch.ones(x.shape[0], 1, device=self.device)
                probs = torch.sigmoid(self.model(x, mask)[0]).float().cpu().numpy()
                for r in range(probs.shape[0]):
                    hit = np.nonzero(probs[r] >= self._thr_vec)[0]
                    if len(hit) > max_per_sent:                        # keep the highest probs
                        hit = hit[np.argsort(-probs[r][hit])[:max_per_sent]]
                    for j in hit:
                        lab, p, s = self.labels[j], float(probs[r, j]), i + r
                        if lab not in best:
                            best[lab] = (p, s)
                        elif p > best[lab][0]:
                            best[lab] = (p, best[lab][1])              # keep the FIRST sent_idx
        top = sorted(best.items(), key=lambda kv: -kv[1][0])[:max_per_doc]
        ttps = [(lab, p) for lab, (p, _s) in top]
        seq = sorted(((s, lab) for lab, (_p, s) in top), key=lambda t: (t[0], t[1]))
        return {"ttps": ttps, "seq": seq}

    def extract(self, text: str, sents=None, emb: torch.Tensor | None = None,
                mode: str = "sentence") -> dict:
        from .utils import sent_split
        sents = (sents if sents is not None else sent_split(text))[: int(self.cfg["max_sents"])]
        if not sents:
            return {"ttps": [], "seq": [], "n_sents": 0}
        if mode == "document":
            out = self.extract_from_emb(emb if emb is not None else self.encoder.encode(sents))
        else:
            out = self.extract_per_sentence(sents, emb)
        out["n_sents"] = len(sents)
        return out


def cmd_extract(args):
    svc = TTPService(config=args.config, device=args.device)
    text = args.text or open(args.file, encoding="utf-8").read()
    out = svc.extract(text)
    print(f"{len(out['ttps'])} TTPs above threshold:")
    for lab, p in out["ttps"][:25]:
        print(f"  {lab:<12} {p:.3f}")
    print("seq (Eq.6):", out["seq"][:15])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["encode", "train", "eval", "extract", "stats"])
    ap.add_argument("--config", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--text", default=None)
    ap.add_argument("--file", default=None)
    args = ap.parse_args()
    {"encode": cmd_encode, "train": cmd_train, "eval": cmd_eval,
     "extract": cmd_extract, "stats": cmd_stats}[args.cmd](args)


if __name__ == "__main__":
    main()
