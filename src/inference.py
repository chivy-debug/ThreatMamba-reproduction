"""Stage 7 inference path: one CTI report -> the complete result payload for the UI.

    text -> IOCs (+ optional live enrichment) -> TTPs (frozen) -> CSKG -> classifier (frozen)
         -> top-k attacker groups + probabilities + node contribution scores (Eq. 21-23)
         + pie-node graph data

Shared by the Streamlit app and the CLI:
    python -m src.inference --file report.txt
    python -m src.inference --doc d00042 --keep 0.6
"""
import argparse
import hashlib
import json
import time

import numpy as np
import torch

from .cskg_builder import NODE_TYPES, REL, GraphBuilder, truncate_graph
from .encoder import get_encoder
from .evaluate import load_model
from .explain import node_contributions
from .ioc_extract import extract_iocs
from .train import HUNT_RELS
from .utils import (ATTCK_DIR, ENRICHED, PROCESSED, load_config, load_json, read_jsonl,
                    sent_split)

STATE_T = NODE_TYPES.index("State")
TECH_T = NODE_TYPES.index("Technique")
TACTIC_T = NODE_TYPES.index("Tactic")
IOC_TYPES = {i for i, t in enumerate(NODE_TYPES) if t not in ("State", "Technique", "Tactic")}
ENRICH_RELS = {REL["rapiddns_history"], REL["otx_general"], REL["vt_object"], REL["vt_behavior"]}


# --------------------------------------------------------------------- live enrichment
def live_enrich(iocs: list[dict], sents: list[str], cfg, max_iocs: int = 3,
                progress=None) -> dict:
    """Trimmed-down IOCHunter for live demos. Consumes VirusTotal quota, so the UI must
    warn before calling it. Returns a dict compatible with GraphBuilder(mode='enriched')."""
    from .ioc_hunter.apis import OTXClient, QuotaError, RapidDNSClient, VTClient, call_method
    from .ioc_hunter.llm_agent import OllamaAgent
    from .ioc_hunter.uct import UCTSelector

    icfg = cfg["iochunter"]
    vt = VTClient(sleep_seconds=float(icfg["vt_sleep_seconds"]))
    otx, rdns = OTXClient(), RapidDNSClient()
    llm = OllamaAgent(model=icfg["llm"], max_retries=int(icfg["max_llm_retries"]))
    uct = UCTSelector()
    accept = float(icfg.get("accept_score", 6))

    huntable = [i for i in iocs if i["type"] in ("Domain", "IP", "URL", "Hash")][:max_iocs]
    hunts, logs = [], []
    for ioc in huntable:
        tried = set()
        for _ in range(int(icfg["max_rounds"])):
            method = uct.select(ioc["type"], exclude=tried)
            if method is None:
                break
            tried.add(method)
            if progress:
                progress(f"{method} <- {ioc['value'][:40]}")
            try:
                cands = call_method(method, ioc, vt, otx, rdns)
            except QuotaError as e:
                logs.append(f"QUOTA {method} {ioc['value']}: {e}")
                uct.update(method, 0)
                continue
            except Exception as e:  # noqa: BLE001
                logs.append(f"ERROR {method} {ioc['value']}: {type(e).__name__}: {e}")
                uct.update(method, 0)
                continue
            direct = [c for c in cands if c["type"] in ("Time", "Geo-location", "CMD", "API")]
            screen = [c for c in cands if c not in direct]
            acc = list(direct)
            if screen:
                ctx = sents[ioc["sent_idx"]] if ioc["sent_idx"] < len(sents) else ""
                try:
                    acc += [c for c in llm.screen(ioc, screen, ctx, method) if c["score"] >= accept]
                except Exception as e:  # noqa: BLE001
                    logs.append(f"LLM-FAIL {method}: {e}")
            uct.update(method, len(acc))
            hunts.append({"seed": ioc, "method": method, "n_candidates": len(cands), "accepted": acc})
    return {"hunts": hunts, "logs": logs}


# --------------------------------------------------------------------- service
class InferenceService:
    def __init__(self, tag: str = "main", config=None, device=None, ttp: bool = True):
        self.cfg = load_config(config)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model, self.groups, self.flags = load_model(tag, self.cfg, self.device)
        self.drop_rels = HUNT_RELS if self.flags["no_iochunter"] else None
        self.encoder = get_encoder(device=str(self.device))
        self.ttp_svc = None
        if ttp:
            from .utils import OUTPUTS
            if (OUTPUTS / "ttp_extractor.pt").exists():
                from .ttp_extract import TTPService
                self.ttp_svc = TTPService(config=config, device=str(self.device))
        self.builder = GraphBuilder(
            encoder=self.encoder,
            ttp_fn=(lambda t, s, e=None: self.ttp_svc.extract(t, s, e)) if self.ttp_svc else None,
            mode="train")
        a = ATTCK_DIR / "attck_v14.json"
        self.attck = load_json(a) if a.exists() else {"techniques": {}, "groups": {}}

    # ---- technique name lookup for display
    def tech_name(self, tid: str) -> str:
        return self.attck.get("techniques", {}).get(tid, {}).get("name", "")

    def predict_graph(self, g: dict) -> tuple[np.ndarray, np.ndarray]:
        """Returns (per-class probabilities, per-node contribution scores)."""
        from .model import collate
        with torch.no_grad():
            logits, _ = self.model(collate([g], self.device, self.drop_rels))
            probs = torch.sigmoid(logits)[0].cpu().numpy()
        _pred, contrib = node_contributions(self.model, g, self.device, self.drop_rels)
        return probs, contrib

    def analyze(self, text: str, doc_id: str = "live", group: str = "?",
                keep_frac: float = 1.0, enriched: dict | None = None,
                progress=None) -> dict:
        t0 = time.time()
        steps = []
        sents = sent_split(text)
        if progress:
            progress("1/4 Extract IOCs")
        iocs = extract_iocs(text, sents)
        steps.append(("IOC", f"{len(iocs)} IOCs from {len(sents)} sentences"))

        if progress:
            progress("2/4 Extract TTPs")
        if self.ttp_svc:
            ttp = self.ttp_svc.extract(text, sents)
            th = self.ttp_svc.thr        # thresholds tuned on the Stage 3 validation split
            steps.append(("TTP", f"{len(ttp['ttps'])} TTPs "
                                 f"(thresholds: tactics {th.get('tactics', 0.5):.2f} / "
                                 f"techniques {th.get('techniques', 0.5):.2f})"))
        else:
            ttp = {"ttps": [], "seq": []}
            steps.append(("TTP", "SKIPPED - outputs/ttp_extractor.pt not found (Stage 3)"))

        if progress:
            progress("3/4 Build CSKG")
        self.builder.mode = "enriched" if enriched else "train"
        g_full = self.builder.build({"doc_id": doc_id, "group": group, "text": text}, enriched)
        g = truncate_graph(g_full, keep_frac) if keep_frac < 1.0 else g_full
        steps.append(("CSKG", f"{len(g['node_types'])} nodes / {len(g['edges'])} edges / "
                              f"{len(g['states'])} states"))

        if progress:
            progress("4/4 Predict")
        probs, contrib = self.predict_graph(g)
        order = np.argsort(-probs)
        topk = [{"group": self.groups[i], "prob": float(probs[i])} for i in order[:5]]
        steps.append(("Predict", f"top-1: {topk[0]['group']} ({topk[0]['prob']:.3f})"))

        return {
            "doc_id": doc_id, "group_true": group, "keep_frac": keep_frac,
            "n_sents": len(sents), "sents": sents,
            "iocs": self._ioc_rows(g, iocs, enriched),
            "ttps": [{"id": t, "name": self.tech_name(t), "prob": round(p, 4),
                      "is_tactic": t.startswith("TA")} for t, p in ttp["ttps"]],
            "topk": topk, "probs": probs.tolist(), "steps": steps,
            "graph": g, "contrib": contrib.tolist(),
            "top_nodes": self._top_nodes(g, contrib),
            "echarts": echarts_payload(g, contrib, topk),
            "elapsed": round(time.time() - t0, 2),
        }

    def _ioc_rows(self, g: dict, iocs: list[dict], enriched) -> list[dict]:
        """IOC table: value, ontology node type, and source (text / hunting function name)."""
        src = {}
        if enriched:
            for h in enriched.get("hunts", []):
                for c in h.get("accepted", []):
                    src[(c["type"], str(c["value"]).lower())] = h["method"]
        rows = [{"value": i["value"], "type": i["type"], "sent": i["sent_idx"], "source": "text"}
                for i in iocs]
        seen = {(r["type"], r["value"].lower()) for r in rows}
        for t, v, s in zip(g["node_types"], g["node_values"], g["node_sent"]):
            key = (NODE_TYPES[t], v.lower())
            if t in IOC_TYPES and key not in seen and key in src:
                rows.append({"value": v, "type": NODE_TYPES[t], "sent": s, "source": src[key]})
        return rows

    @staticmethod
    def _top_nodes(g: dict, contrib: np.ndarray, k: int = 15) -> list[dict]:
        idx = np.argsort(-contrib)[:k]
        return [{"node": g["node_values"][i], "type": NODE_TYPES[g["node_types"][i]],
                 "sent": g["node_sent"][i], "score": round(float(contrib[i]), 5)} for i in idx]


# --------------------------------------------------------------------- ECharts pie-node
def echarts_payload(g: dict, contrib, topk: list[dict], max_states: int = 14,
                    max_pie: int = 8) -> dict:
    """Produce JSON in the format expected by the original authors' pie-node template
    (`pie-node-ttp-state-graph.temp.html`): each State node is a pie chart of TTP
    contribution shares, symbolSize encodes importance and edge opacity encodes
    contribution."""
    contrib = np.asarray(contrib, dtype=float)
    types, values, sent = g["node_types"], g["node_values"], g["node_sent"]

    # Techniques attached to each state:
    #   (a) directly, via a dependency edge (the technique appears in that sentence)
    #   (b) indirectly, via that state's IOCs (ioc_ttp edges). This fills the pies out and
    #       matches the "TTP contribution share at a state" idea of Fig. 6c.
    edge_list = g["edges"].tolist()
    tech_of_state: dict[int, list[int]] = {}
    ioc_of_state: dict[int, list[int]] = {}
    tech_of_ioc: dict[int, list[int]] = {}
    for u, v, r in edge_list:
        if r == REL["dependency"] and types[u] == STATE_T:
            if types[v] == TECH_T:
                tech_of_state.setdefault(u, []).append(v)
            elif types[v] in IOC_TYPES:
                ioc_of_state.setdefault(u, []).append(v)
        elif r == REL["ioc_ttp"] and types[v] == TECH_T:
            tech_of_ioc.setdefault(u, []).append(v)
    for s, iocs_ in ioc_of_state.items():
        for i in iocs_:
            for t in tech_of_ioc.get(i, []):
                tech_of_state.setdefault(s, []).append(t)

    states = [s for s in g["states"]]
    states.sort(key=lambda n: -contrib[n])
    states = sorted(states[:max_states], key=lambda n: sent[n])
    if not states:
        return {"nodes": [], "edges": []}
    cmax = max(float(contrib[s]) for s in states) or 1.0

    nodes = [{"id": "document", "name": "document", "symbolSize": 30, "isPie": False,
              "label": {"show": True}, "symbol": "triangle", "itemStyle": {"color": "#ee6666"}}]
    edges = []
    for gi in topk[:3]:
        nodes.append({"id": gi["group"], "name": gi["group"],
                      "symbolSize": round(12 + 28 * float(gi["prob"]), 2),
                      "isPie": False, "label": {"show": True}})

    for s in states:
        sid = f"state-{sent[s]}"
        agg: dict[str, float] = {}
        for t in set(tech_of_state.get(s, [])):        # dedupe nodes, aggregate by technique
            agg[values[t]] = agg.get(values[t], 0.0) + float(contrib[t])
        pie = [{"value": v, "name": k} for k, v in agg.items()]
        pie.sort(key=lambda d: -d["value"])
        pie = pie[:max_pie]
        tot = sum(d["value"] for d in pie)
        size = round(20 + 130 * float(contrib[s]) / cmax, 2)
        if pie and tot > 0:
            nodes.append({"id": sid, "name": sid, "symbolSize": size, "isPie": True,
                          "pieData": [{"value": round(d["value"] / tot, 4), "name": d["name"]}
                                      for d in pie],
                          "label": {"show": False}})
        else:  # state with no TTPs -> plain circular node
            nodes.append({"id": sid, "name": sid, "symbolSize": round(size / 3, 2),
                          "isPie": False, "label": {"show": True}})
        edges.append({"source": "document", "target": sid, "edgeLabel": {"show": False}})
        for gi in topk[:3]:
            score = round(float(contrib[s]) * float(gi["prob"]), 4)
            edges.append({"source": sid, "target": gi["group"],
                          "lineStyle": {"opacity": round(min(0.9, float(contrib[s]) / cmax), 4)},
                          "score": score,
                          "label": {"show": True, "formatter": "{@score}"}})
    return {"nodes": nodes, "edges": edges}


# Recolour the template for the dark theme. The authors' original file is NEVER modified;
# the substitutions happen at render time, so the template on disk stays pristine and
# remains directly comparable with the upstream version.
_DARK_SWAP = [
    ("background: white;", "background: #0f151c;"),
    ("background: #fff;", "background: #0f151c;"),
    ("color: #333;", "color: #e6edf3;"),
    ("background: #e3f2fd;", "background: #14212e;"),
    ("color: #1976d2;", "color: #7cc6ff;"),
    ("ctx.fillStyle = 'white';", "ctx.fillStyle = '#0f151c';"),
    ("ctx.fillStyle = 'black';", "ctx.fillStyle = '#e6edf3';"),
    ("ctx.fillStyle = 'rgba(255, 255, 255, 0.95)';", "ctx.fillStyle = 'rgba(15, 21, 28, 0.94)';"),
    ("ctx.strokeStyle = '#ddd';", "ctx.strokeStyle = '#4b5b6e';"),
    ("ctx.fillStyle = '#333';", "ctx.fillStyle = '#e6edf3';"),
    ("color: '#333'", "color: '#e6edf3'"),      # also catches the trailing-comma variant
    ("color: '#999',", "color: '#4b5b6e',"),
]


def render_pie_html(payload: dict, template_path=None, height: int = 720,
                    dark: bool = False) -> str:
    """Inject the payload into the authors' template, returning a self-contained HTML
    document (embedded via components.html)."""
    from pathlib import Path
    tpl = Path(template_path) if template_path else Path(__file__).resolve().parents[1] / "app" / "pie_node_graph.html"
    html = tpl.read_text(encoding="utf-8")
    if dark:
        for a, b in _DARK_SWAP:
            html = html.replace(a, b)
    # ascii-only + escaping for a JS string literal inside a <script> tag:
    #   \ -> \\ , ' -> \' , </ -> <\/ (so the tag cannot be closed early by the data)
    data = (json.dumps(payload, ensure_ascii=True)
            .replace("\\", "\\\\").replace("'", "\\'").replace("</", "<\\/"))
    html = html.replace("__GRAPH_DATA__", data).replace("__CHART_HEIGHT__", str(height))
    return html


# --------------------------------------------------------------------- CLI
def load_demo_doc(doc_id: str) -> dict:
    for d in read_jsonl(PROCESSED / "docs.jsonl"):
        if d["doc_id"] == doc_id:
            return d
    raise SystemExit(f"document {doc_id} not found")


def load_enriched(doc_id: str):
    fp = ENRICHED / f"{doc_id}.json"
    return json.load(open(fp, encoding="utf-8")) if fp.exists() else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default=None)
    ap.add_argument("--text", default=None)
    ap.add_argument("--doc", default=None, help="doc_id from the processed dataset")
    ap.add_argument("--keep", type=float, default=1.0,
                    help="fraction of the timeline to keep (robustness)")
    ap.add_argument("--enriched", action="store_true",
                    help="use the document's enriched file when available")
    ap.add_argument("--tag", default="main")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    if args.doc:
        d = load_demo_doc(args.doc)
        text, doc_id, group = d["text"], d["doc_id"], d["group"]
    else:
        text = args.text or open(args.file, encoding="utf-8").read()
        doc_id, group = "live", "?"
    svc = InferenceService(tag=args.tag, config=args.config)
    res = svc.analyze(text, doc_id, group, keep_frac=args.keep,
                      enriched=load_enriched(doc_id) if args.enriched else None)
    print(f"--- {doc_id} (ground truth: {group}) | {res['elapsed']}s ---")
    for name, info in res["steps"]:
        print(f"  {name:<8} {info}")
    print("Top-5 attacker:")
    for r in res["topk"]:
        print(f"  {r['prob']:.4f}  {r['group']}")
    print("Highest-contributing nodes:")
    for n in res["top_nodes"][:8]:
        print(f"  {n['score']:.4f}  {n['type']:<12} {n['node'][:50]}")


if __name__ == "__main__":
    main()
