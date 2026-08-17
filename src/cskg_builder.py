"""Stage 4: build the time-ordered CSKG with the full 19-type node ontology (Fig. 3)
plus the State node.

Node types (19 + State):
  State | Domain URL IP Email Hash Filename CVE Account Port Protocol MAC Registry Header
  | Time Geo-location CMD API (from IOCHunter) | Technique Tactic

Relations (edge labels):
  temporal_next   : State_i -> State_{i+1} (the timeline backbone, Eq. 6)
  dependency      : State <-> IOC/TTP occurring in that sentence
  co_occurs       : IOC-IOC co-occurrence within a sentence (train layer)
  rapiddns_history/otx_general/vt_object/vt_behavior : hunting-function edges (enriched layer)
  ioc_ttp         : IOC -> the 3 nearest Techniques in the text
  subsumption     : Technique -> Tactic (ATT&CK)
  tactic_seq      : Tactic -> Tactic in ATT&CK kill-chain order
  tech_temporal   : Technique -> Technique in order of appearance

Serialised to data/cskg/{doc_id}.pt as
  {doc_id, group, node_types, node_values, node_sent,
   x (float16 N,768), edges LongTensor(E,3), states [list]}

CLI:
  python -m src.cskg_builder build [--split all|train|test] [--mode train|enriched]
                                   [--ttp auto|none]
  python -m src.cskg_builder stats
  python -m src.cskg_builder render --doc d00042
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import torch

from .encoder import encode_doc_cached, get_encoder
from .ioc_extract import extract_iocs
from .utils import (ATTCK_DIR, CSKG_DIR, ENRICHED, OUTPUTS, load_config, load_json, read_jsonl,
                    sent_split, torch_load)
from .utils import PROCESSED

NODE_TYPES = ["State", "Domain", "URL", "IP", "Email", "Hash", "Filename", "CVE", "Account",
              "Port", "Protocol", "MAC", "Registry", "Header", "Time", "Geo-location",
              "CMD", "API", "Technique", "Tactic"]
NT = {t: i for i, t in enumerate(NODE_TYPES)}

RELATIONS = ["temporal_next", "dependency", "co_occurs", "rapiddns_history", "otx_general",
             "vt_object", "vt_behavior", "ioc_ttp", "subsumption", "tactic_seq", "tech_temporal"]
REL = {r: i for i, r in enumerate(RELATIONS)}


def _attck():
    fp = ATTCK_DIR / "attck_v14.json"
    return load_json(fp) if fp.exists() else {"techniques": {}, "tactic_order": []}


class GraphBuilder:
    def __init__(self, encoder=None, ttp_fn=None, mode: str = "train", max_sents: int = 256):
        """ttp_fn(text, sents, emb) -> {"ttps": [(label, prob)], "seq": [(sent_idx, label)]}
        or None. `emb` is the cached sentence embedding, passed through so the TTP module
        does not have to encode the document a second time."""
        self.encoder = encoder or get_encoder()
        self.ttp_fn = ttp_fn
        self.mode = mode
        self.max_sents = max_sents
        a = _attck()
        self.tech2tac = {tid: v["tactics"] for tid, v in a["techniques"].items()}
        self.tac_order = [t["shortname"] for t in a["tactic_order"]]
        self.tacid2short = {t["id"]: t["shortname"] for t in a["tactic_order"]}

    def build(self, doc: dict, enriched: dict | None = None) -> dict:
        text = doc["text"]
        sents = sent_split(text)[: self.max_sents]
        emb = encode_doc_cached(doc["doc_id"], sents, self.encoder).float()
        iocs = extract_iocs(text, sents)
        ttp = self.ttp_fn(text, sents, emb) if self.ttp_fn else {"ttps": [], "seq": []}

        nodes: list[tuple[int, str, int]] = []   # (type_id, value, first_sentence_index)
        feats: list[torch.Tensor] = []
        index: dict[tuple[int, str], int] = {}
        edges: list[tuple[int, int, int]] = []
        doc_mean = emb.mean(0) if len(emb) else torch.zeros(768)

        def add_node(t: str, value: str, sent_idx: int, feat: torch.Tensor) -> int:
            key = (NT[t], value.lower())
            if key in index:
                return index[key]
            index[key] = len(nodes)
            nodes.append((NT[t], value, sent_idx))
            feats.append(feat)
            return index[key]

        def add_edge(u: int, v: int, r: str):
            e = (u, v, REL[r])
            edges.append(e)

        # ---- states: sentences containing at least one IOC or TTP
        active = sorted({i["sent_idx"] for i in iocs} | {s for s, _l in ttp["seq"]})
        active = [s for s in active if s < len(sents)]
        if not active:
            active = [0] if sents else []
        states = []
        for si in active:
            feat = emb[si] if si < len(emb) else doc_mean
            states.append(add_node("State", f"state_{si}", si, feat))
        for a_, b_ in zip(states, states[1:]):
            add_edge(a_, b_, "temporal_next")
        state_of_sent = dict(zip(active, states))

        # ---- IOC nodes
        ioc_ids = []
        for i in iocs:
            si = min(i["sent_idx"], max(0, len(emb) - 1))
            feat = emb[si] if len(emb) else doc_mean
            nid = add_node(i["type"], i["value"], i["sent_idx"], feat)
            ioc_ids.append((nid, i))
            st = state_of_sent.get(i["sent_idx"])
            if st is not None:
                add_edge(st, nid, "dependency")

        # ---- co-occurrence (train layer): IOCs sharing a sentence
        by_sent: dict[int, list[int]] = {}
        for nid, i in ioc_ids:
            by_sent.setdefault(i["sent_idx"], []).append(nid)
        for _si, ids in by_sent.items():
            ids = list(dict.fromkeys(ids))[:6]
            for x in range(len(ids)):
                for y in range(x + 1, len(ids)):
                    add_edge(ids[x], ids[y], "co_occurs")

        # ---- TTP nodes (Technique/Tactic)
        tech_first: dict[str, int] = {}
        for si, lab in ttp["seq"]:
            if lab.startswith("TA"):
                continue
            tech_first.setdefault(lab, si)
        tech_nodes: dict[str, int] = {}
        for lab, si in tech_first.items():
            feat = emb[si] if si < len(emb) else doc_mean
            nid = add_node("Technique", lab, si, feat)
            tech_nodes[lab] = nid
            st = state_of_sent.get(si)
            if st is not None:
                add_edge(st, nid, "dependency")
        # tactic nodes: directly predicted (TA...) plus the parents of predicted techniques
        tac_short_present: dict[str, int] = {}
        for si, lab in ttp["seq"]:
            if lab.startswith("TA"):
                short = self.tacid2short.get(lab, lab)
                tac_short_present.setdefault(short, si)
        for lab, nid in tech_nodes.items():
            base = lab.split(".")[0]
            for short in self.tech2tac.get(lab, self.tech2tac.get(base, [])):
                tac_short_present.setdefault(short, tech_first[lab])
        tac_nodes: dict[str, int] = {}
        for short, si in tac_short_present.items():
            feat = emb[si] if si < len(emb) else doc_mean
            tac_nodes[short] = add_node("Tactic", short, si, feat)
            st = state_of_sent.get(si)
            if st is not None:
                add_edge(st, tac_nodes[short], "dependency")
        # subsumption + tactic_seq + tech_temporal
        for lab, nid in tech_nodes.items():
            base = lab.split(".")[0]
            for short in self.tech2tac.get(lab, self.tech2tac.get(base, [])):
                if short in tac_nodes:
                    add_edge(nid, tac_nodes[short], "subsumption")
        chain = [s for s in self.tac_order if s in tac_nodes]
        for a_, b_ in zip(chain, chain[1:]):
            add_edge(tac_nodes[a_], tac_nodes[b_], "tactic_seq")
        order = sorted(tech_nodes, key=lambda l: (tech_first[l], l))
        for a_, b_ in zip(order, order[1:]):
            add_edge(tech_nodes[a_], tech_nodes[b_], "tech_temporal")
        # ioc_ttp: link each IOC to at most the 3 nearest techniques in the text
        for nid, i in ioc_ids:
            near = sorted(tech_nodes, key=lambda l: (abs(tech_first[l] - i["sent_idx"]), l))[:3]
            for l in near:
                add_edge(nid, tech_nodes[l], "ioc_ttp")

        # ---- enriched layer (Stage 2), only in enriched mode
        if self.mode == "enriched" and enriched:
            for hunt in enriched.get("hunts", []):
                seed = hunt["seed"]
                skey = (NT.get(seed["type"], -1), seed["value"].lower())
                if skey not in index:
                    continue
                sid = index[skey]
                for c in hunt.get("accepted", []):
                    if c["type"] not in NT:
                        continue
                    cid = add_node(c["type"], str(c["value"]), nodes[sid][2], feats[sid])
                    add_edge(sid, cid, hunt["method"])

        N = len(nodes)
        x = torch.stack(feats) if feats else torch.zeros(0, 768)
        g = {"doc_id": doc["doc_id"], "group": doc.get("group", "?"),
             "node_types": [n[0] for n in nodes], "node_values": [n[1] for n in nodes],
             "node_sent": [n[2] for n in nodes], "x": x.half(),
             "edges": torch.tensor(edges, dtype=torch.long) if edges else torch.zeros(0, 3, dtype=torch.long),
             "states": states, "n_sents": len(sents)}
        assert len(g["node_types"]) == N == g["x"].shape[0]
        return g


def truncate_graph(g: dict, keep_frac: float) -> dict:
    """Robustness masking (Table VIII): keep the first k% of the timeline BY STATE ORDER,
    drop nodes whose node_sent falls past the cut-off, and keep edges between survivors."""
    states = g["states"]
    if not states or keep_frac >= 1.0:
        return g
    k = max(1, int(round(len(states) * keep_frac)))
    max_sent = g["node_sent"][states[k - 1]]
    keep = [i for i in range(len(g["node_types"])) if g["node_sent"][i] <= max_sent]
    remap = {old: new for new, old in enumerate(keep)}
    keepset = set(keep)
    edges = [(remap[u], remap[v], r) for u, v, r in g["edges"].tolist()
             if u in keepset and v in keepset]
    return {**g,
            "node_types": [g["node_types"][i] for i in keep],
            "node_values": [g["node_values"][i] for i in keep],
            "node_sent": [g["node_sent"][i] for i in keep],
            "x": g["x"][keep],
            "edges": torch.tensor(edges, dtype=torch.long) if edges else torch.zeros(0, 3, dtype=torch.long),
            "states": [remap[s] for s in states[:k] if s in keepset]}


# ------------------------------------------------------------------ CLI
def _ttp_fn(args, cfg):
    if args.ttp == "none":
        return None
    ckpt = OUTPUTS / "ttp_extractor.pt"
    if not ckpt.exists():
        print("[cskg] WARNING: ttp_extractor.pt not found -> building CSKGs WITHOUT TTP nodes "
              "(run Stage 3 training first for complete graphs)")
        return None
    from .ttp_extract import TTPService
    svc = TTPService(config=args.config)
    return lambda text, sents, emb=None: svc.extract(text, sents, emb)


def cmd_build(args):
    cfg = load_config(args.config)
    docs = read_jsonl(PROCESSED / "docs.jsonl")
    if args.split != "all":
        docs = [d for d in docs if d["split"] == args.split]
    if args.limit:
        docs = docs[: args.limit]
    builder = GraphBuilder(ttp_fn=_ttp_fn(args, cfg), mode=args.mode)
    CSKG_DIR.mkdir(parents=True, exist_ok=True)
    out_dir = CSKG_DIR if args.mode == "train" else CSKG_DIR / "enriched"
    out_dir.mkdir(parents=True, exist_ok=True)
    n_nodes = n_edges = 0
    for k, doc in enumerate(docs):
        enr = None
        if args.mode == "enriched":
            fp = ENRICHED / f"{doc['doc_id']}.json"
            enr = json.load(open(fp, encoding="utf-8")) if fp.exists() else None
        g = builder.build(doc, enr)
        torch.save(g, out_dir / f"{doc['doc_id']}.pt")
        n_nodes += len(g["node_types"]); n_edges += len(g["edges"])
        if (k + 1) % 200 == 0:
            print(f"  built {k + 1}/{len(docs)}")
    print(f"OK: {len(docs)} CSKG ({args.mode}) -> {out_dir} | "
          f"mean {n_nodes / max(1, len(docs)):.1f} nodes / {n_edges / max(1, len(docs)):.1f} edges")


def cmd_stats(args):
    files = sorted(CSKG_DIR.glob("d*.pt"))
    if not files:
        raise SystemExit("no CSKG found - run `build` first")
    cn, ce, ctype, crel = [], [], Counter(), Counter()
    for fp in files:
        g = torch_load(fp)
        cn.append(len(g["node_types"])); ce.append(len(g["edges"]))
        ctype.update(NODE_TYPES[t] for t in g["node_types"])
        crel.update(RELATIONS[r] for _u, _v, r in g["edges"].tolist())
    import numpy as np
    print("=" * 55)
    print(f"STAGE 4 ACCEPTANCE - {len(files)} CSKGs")
    print(f"nodes/graph: mean {np.mean(cn):.1f}  p50 {np.median(cn):.0f}  max {max(cn)}"
          f"   (paper ~53)")
    print(f"edges/graph: mean {np.mean(ce):.1f}  p50 {np.median(ce):.0f}  max {max(ce)}"
          f"   (paper ~147)")
    print("node type distribution:", dict(ctype.most_common()))
    print("relation distribution:", dict(crel.most_common()))
    print(f"node types covered: {len(ctype)}/20 (including State; the full 19-type ontology "
          f"needs the Stage 2 daemon)")


def cmd_render(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import networkx as nx
    fp = (CSKG_DIR / "enriched" / f"{args.doc}.pt")
    if not fp.exists():
        fp = CSKG_DIR / f"{args.doc}.pt"
    g = torch_load(fp)
    G = nx.DiGraph()
    for i, (t, v) in enumerate(zip(g["node_types"], g["node_values"])):
        G.add_node(i, label=f"{NODE_TYPES[t]}:{v[:18]}", t=t)
    for u, v, r in g["edges"].tolist():
        G.add_edge(u, v, r=r)
    cmap = plt.cm.tab20
    colors = [cmap(t % 20) for t in g["node_types"]]
    plt.figure(figsize=(14, 10))
    pos = nx.spring_layout(G, seed=42, k=0.6)
    nx.draw(G, pos, node_color=colors, node_size=240, arrows=True, width=0.5, alpha=0.9)
    nx.draw_networkx_labels(G, pos, {i: d["label"] for i, d in G.nodes(data=True)}, font_size=5)
    out = OUTPUTS / f"cskg_{args.doc}.png"
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(); plt.savefig(out, dpi=180); plt.close()
    print(f"OK: {out} | {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "stats", "render"])
    ap.add_argument("--split", default="all", choices=["all", "train", "test"])
    ap.add_argument("--mode", default="train", choices=["train", "enriched"])
    ap.add_argument("--ttp", default="auto", choices=["auto", "none"])
    ap.add_argument("--doc", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    {"build": cmd_build, "stats": cmd_stats, "render": cmd_render}[args.cmd](args)


if __name__ == "__main__":
    main()
