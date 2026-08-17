"""Stage 7 demo UI for ThreatMAMBA (Streamlit, four pages).

Dark SOC theme. Colours and CSS live in app/ui_kit.py and .streamlit/config.toml.

Run:  cd <repo root> && streamlit run app/streamlit_app.py
      (it MUST be run from the repository root so Streamlit picks up .streamlit/config.toml)
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "app"))

from src.inference import InferenceService, live_enrich, render_pie_html   # noqa: E402
from src.utils import ENRICHED, OUTPUTS, PROCESSED, load_config, read_jsonl  # noqa: E402
from ui_kit import (CSS, badge, kpi_row, node_badge, note, page_head,        # noqa: E402
                    rank_list, section, stat_line, steps_bar, verdict)

st.set_page_config(page_title="ThreatMAMBA — Cyber Threat Attribution",
                   page_icon="🛡️", layout="wide", initial_sidebar_state="expanded")
st.markdown(CSS, unsafe_allow_html=True)

# --- Streamlit API compatibility (new: width="stretch" + st.iframe; old: use_container_width)
_NEW_API = hasattr(st, "iframe")
STRETCH = {"width": "stretch"} if _NEW_API else {"use_container_width": True}


def embed_html(html: str, height: int):
    """Embed HTML that contains JavaScript (ECharts)."""
    if _NEW_API:
        st.iframe(html, height=height)
    else:
        st.components.v1.html(html, height=height, scrolling=False)


def H(html: str):
    st.markdown(html, unsafe_allow_html=True)


# --------------------------------------------------------------- resource loading
@st.cache_resource(show_spinner="Loading models (SecureBERT + TTP + classifier)…")
def get_service(tag: str):
    return InferenceService(tag=tag)


@st.cache_data(show_spinner=False)
def get_docs():
    fp = PROCESSED / "docs.jsonl"
    return {d["doc_id"]: d for d in read_jsonl(fp)} if fp.exists() else {}


@st.cache_data(show_spinner=False)
def get_demo_subset():
    fp = PROCESSED.parent / "demo_subset.txt"
    return [l.strip() for l in fp.read_text(encoding="utf-8").splitlines()
            if l.strip() and not l.startswith("#")] if fp.exists() else []


@st.cache_data(show_spinner=False, max_entries=64)
def run_analysis(text: str, doc_id: str, group: str, keep: float, tag: str, enriched_json: str | None):
    """Cached on (content, timeline cut-off) so the page-2 slider responds instantly."""
    svc = get_service(tag)
    enr = json.loads(enriched_json) if enriched_json else None
    res = svc.analyze(text, doc_id=doc_id, group=group, keep_frac=keep, enriched=enr)
    res.pop("graph", None)      # tensors are not needed by the UI; drop them to keep the cache small
    return res


@st.cache_data(show_spinner=False)
def acceptance_main():
    """Read metrics_all.csv -> (the 'main' row, pass/fail). None when it does not exist yet."""
    fp = OUTPUTS / "metrics_all.csv"
    if not fp.exists():
        return None
    df = pd.read_csv(fp)
    row = df[df["model"] == "main"]
    if not len(row):
        return None
    r = row.iloc[0].to_dict()
    return r, (float(r["f1_macro"]) > 0.35 and float(r["top3_micro"]) > 0.60)


def enriched_of(doc_id: str):
    fp = ENRICHED / f"{doc_id}.json"
    return fp.read_text(encoding="utf-8") if fp.exists() else None


def need(msg_html: str):
    H(note(msg_html))
    st.stop()


def check_ready():
    if not (OUTPUTS / "model_main.pt").exists():
        need("<b>outputs/model_main.pt</b> is missing. Run Stage 5 first:<br>"
             "<code>bash scripts/train_all.sh</code>")


# ------------------------------------------------------------------------ sidebar
PAGES = ["CTI report analysis", "Robustness over time",
         "Threat group profile", "Consolidated results"]
PAGE_ICONS = ["◈", "◧", "◎", "▤"]

with st.sidebar:
    H('<div class="brand"><div class="brand-mark">🛡️</div>'
      '<div><div class="brand-name">ThreatMAMBA</div>'
      '<div class="brand-sub">Cyber Threat Attribution</div></div></div>')

    page = st.radio("Page", PAGES, label_visibility="collapsed",
                    format_func=lambda p: f"{PAGE_ICONS[PAGES.index(p)]}  {p}")

    tags = [p.stem.replace("model_", "") for p in sorted(OUTPUTS.glob("model_*.pt"))] or ["main"]
    st.markdown("---")
    tag = st.selectbox("Model", tags,
                       index=tags.index("main") if "main" in tags else 0,
                       help="main = the full model. Other tags are ablations with one "
                            "component removed.")

    # ---- system status ----
    import torch
    n_enr = len(list(ENRICHED.glob("*.json"))) if ENRICHED.exists() else 0
    acc = acceptance_main()
    _docs = get_docs()
    checks = [
        ("Device", "GPU (CUDA)" if torch.cuda.is_available() else "CPU",
         "ok" if torch.cuda.is_available() else "wa"),
        ("Stage 1 data", f"{len(_docs)} documents", "ok" if _docs else "no"),
        ("TTP module", "ready" if (OUTPUTS / "ttp_extractor.pt").exists() else "missing",
         "ok" if (OUTPUTS / "ttp_extractor.pt").exists() else "no"),
        ("Classifier", f"{len(tags)} checkpoint(s)",
         "ok" if (OUTPUTS / "model_main.pt").exists() else "no"),
        ("Enrichment (Stage 2)", f"{n_enr} documents", "ok" if n_enr >= 5 else "wa"),
        ("Stage 5 acceptance", "PASS" if acc and acc[1] else ("FAIL" if acc else "not run"),
         "ok" if acc and acc[1] else ("no" if acc else "wa")),
    ]
    with st.expander("System status", expanded=True):
        H("".join(stat_line(k, v, s) for k, v, s in checks))

    st.caption("Reproduction of ThreatMAMBA (IEEE TIFS 2026).")


# ================================================================= PAGE 1
def page_analyze():
    H(page_head(
        "CTI report analysis",
        "A single English-language Cyber Threat Intelligence report is run through the whole "
        "pipeline: IOC extraction → TTP extraction → CSKG construction → threat group "
        "attribution, together with the contribution score of every node."))
    check_ready()
    docs, subset = get_docs(), get_demo_subset()

    H(section("Input source"))
    col1, col2 = st.columns([3, 2])
    with col1:
        mode = st.radio("Source", ["Pick from the demo set", "Paste text"],
                        horizontal=True, label_visibility="collapsed")
        text, doc_id, group, enr = "", "live", "?", None
        if mode == "Pick from the demo set":
            if not docs:
                need("<b>data/processed/docs.jsonl</b> is missing. Run Stage 1:<br>"
                     "<code>python -m src.data_prep all</code>")
            ids = subset or sorted(docs)[:50]
            has_enr = {i for i in ids if (ENRICHED / f"{i}.json").exists()}
            doc_id = st.selectbox(
                "Document", ids,
                format_func=lambda i: f"{i} · {docs[i]['group'] if i in docs else '?'}"
                                      f"{'  ✓ enriched' if i in has_enr else ''}")
            if doc_id in docs:
                text, group = docs[doc_id]["text"], docs[doc_id]["group"]
            if doc_id in has_enr and st.checkbox(
                    "Use enrichment data (collected by the IOCHunter daemon)", True):
                enr = enriched_of(doc_id)
        else:
            text = st.text_area("Report text", height=200, label_visibility="collapsed",
                                placeholder="Paste the full English CTI report here…")
            if st.checkbox("Live enrichment — call VirusTotal / OTX / RapidDNS + Qwen3-8B for real"):
                H(note("This consumes VirusTotal quota (4 requests per minute) and takes tens "
                       "of seconds. Only enable it during a live demonstration."))
                if text.strip() and st.button("Run IOCHunter now", type="primary"):
                    from src.ioc_extract import extract_iocs
                    from src.utils import sent_split
                    sents = sent_split(text)
                    box = st.empty()
                    out = live_enrich(extract_iocs(text, sents), sents, load_config(),
                                      max_iocs=3, progress=lambda m: box.info(m, icon="🔎"))
                    st.session_state["live_enr"] = json.dumps(out)
                    box.success(f"Done: {len(out['hunts'])} hunting rounds, "
                                f"{sum(len(h['accepted']) for h in out['hunts'])} new nodes")
                enr = st.session_state.get("live_enr")

    with col2:
        if text.strip():
            H(kpi_row([("Characters", f"{len(text):,}"), ("Words", f"{len(text.split()):,}"),
                       ("Ground truth", group if group != "?" else "—")]))
            with st.expander("Preview the text"):
                st.text(text[:1500] + ("…" if len(text) > 1500 else ""))

    if not text.strip():
        H(note("Pick a document from the demo set, or paste some text, to begin."))
        return

    res = run_analysis(text, doc_id, group, 1.0, tag, enr)
    top1 = res["topk"][0]
    top3_names = [t["group"] for t in res["topk"][:3]]

    # ---------- verdict ----------
    H(section("Attribution result"))
    if res["group_true"] != "?":
        hit = top1["group"] == res["group_true"]
        in3 = res["group_true"] in top3_names
        chip = badge("TOP-1 CORRECT", "green") if hit else (
            badge("TOP-1 WRONG · CORRECT WITHIN TOP-3", "amber") if in3
            else badge("INCORRECT", "red"))
        right = chip + (f'<div class="kpi-s" style="margin-top:.4rem">'
                        f'Ground truth: {res["group_true"]}</div>')
    else:
        right = badge("PASTED TEXT — no ground-truth label", "muted")
    H(verdict("Attributed threat group", f'{top1["group"]}  ·  {top1["prob"]:.3f}', right))

    st.write("")
    H(kpi_row([
        ("Processing time", f'{res["elapsed"]} s'),
        ("Sentences", f'{res["n_sents"]}'),
        ("IOCs", f'{len(res["iocs"])}'),
        ("TTPs", f'{len(res["ttps"])}'),
        ("Graph nodes", dict(res["steps"])["CSKG"].split(" node")[0]),
    ]))

    H(section("Pipeline progress"))
    H(steps_bar(res["steps"]))

    # ---------- top-5 + node contributions ----------
    c1, c2 = st.columns([2, 3])
    with c1:
        H(section("Five most likely groups"))
        H(rank_list(res["topk"]))
        st.caption("Bar length is proportional to each class's sigmoid probability (Eq. 17). "
                   "The probabilities do NOT sum to 1, because the model is trained "
                   "multi-label with BCE.")
    with c2:
        H(section("Nodes contributing most to the decision (Eq. 21-23)"))
        dn = pd.DataFrame(res["top_nodes"])
        if len(dn):
            dn = dn.rename(columns={"node": "Node", "type": "Type",
                                    "sent": "Sentence", "score": "Score"})
            st.dataframe(dn, hide_index=True, height=268, **STRETCH)
        st.caption("Score = |gradient × input| over the node's feature vector, "
                   "normalised to sum to 1.")

    # ---------- TTP / IOC ----------
    H(section("Extracted entities"))
    t1, t2 = st.tabs([f"TTPs  ({len(res['ttps'])})", f"IOCs  ({len(res['iocs'])})"])
    with t1:
        if res["ttps"]:
            d = pd.DataFrame(res["ttps"])
            d["kind"] = np.where(d["is_tactic"], "Tactic", "Technique")
            d = d[["id", "name", "kind", "prob"]].rename(
                columns={"id": "ATT&CK ID", "name": "Name", "kind": "Kind",
                         "prob": "Probability"})
            st.dataframe(d, hide_index=True, height=300, **STRETCH)
            H(" ".join(node_badge("Tactic" if t["is_tactic"] else "Technique") + f" {t['id']}"
                       for t in res["ttps"][:12]))
        else:
            H(note("No TTP passed the threshold, or Stage 3 has not been trained yet "
                   "(<code>python -m src.ttp_extract train</code>)."))
    with t2:
        di = pd.DataFrame(res["iocs"])
        if len(di):
            di = di.rename(columns={"value": "Value", "type": "Type",
                                    "sent": "Sentence", "source": "Source"})
            st.dataframe(di, hide_index=True, height=300, **STRETCH)
            H(" ".join(node_badge(t) for t in sorted({r["type"] for r in res["iocs"]})))
        st.caption("Source column: `text` means the IOC was read straight out of the report; "
                   "any other value is the IOCHunter hunting function that discovered it.")

    # ---------- graph ----------
    H(section("CSKG pie-node graph (rendered with the original authors' template)"))
    if res["echarts"]["nodes"]:
        embed_html(render_pie_html(res["echarts"], height=680, dark=True), height=700)
        st.caption("Each State node is a pie chart showing the contribution share of the "
                   "techniques at that step · node size = importance · edge opacity = "
                   "contribution.")
    else:
        H(note("Empty graph — this document produced no State nodes "
               "(no IOC or TTP could be extracted)."))


# ================================================================= PAGE 2
def page_robustness():
    H(page_head(
        "Robustness — attribution from partial information",
        "The paper progressively masks the timeline from the end, keeping only 20-100% of the "
        "event sequence, and re-measures accuracy (Table VIII, Eq. 27). The question is "
        "whether the system identifies the right group as soon as an investigation begins, "
        "rather than only once all the evidence is in."))
    check_ready()
    docs, subset = get_docs(), get_demo_subset()
    if not docs:
        need("Stage 1 data is missing. Run <code>python -m src.data_prep all</code>")
    ids = subset or sorted(docs)[:50]

    c0, c1 = st.columns([2, 3])
    with c0:
        doc_id = st.selectbox("Document", ids, format_func=lambda i: f"{i} · {docs[i]['group']}")
    d = docs[doc_id]
    with c1:
        keep = st.slider("Percentage of the timeline retained", 20, 100, 100, step=20) / 100

    res = run_analysis(d["text"], doc_id, d["group"], keep, tag, None)
    top1 = res["topk"][0]
    hit = top1["group"] == d["group"]
    in3 = d["group"] in [t["group"] for t in res["topk"][:3]]
    chip = badge("TOP-1 CORRECT", "green") if hit else (
        badge("CORRECT WITHIN TOP-3", "amber") if in3 else badge("INCORRECT", "red"))

    H(verdict(f"Leading group with only {int(keep * 100)}% of the timeline",
              f'{top1["group"]}  ·  {top1["prob"]:.3f}',
              chip + f'<div class="kpi-s" style="margin-top:.4rem">Ground truth: {d["group"]}</div>'))
    st.write("")
    H(kpi_row([("CSKG", dict(res["steps"])["CSKG"]),
               ("Sentences", f'{res["n_sents"]}'),
               ("Processing time", f'{res["elapsed"]} s')]))

    a, b = st.columns([2, 3])
    with a:
        H(section("Ranking at the current cut-off"))
        H(rank_list(res["topk"]))
    with b:
        H(section("Probability across the five timeline cut-offs"))
        with st.spinner("Computing all five cut-offs…"):
            groups_all = get_service(tag).groups
            # Track a FIXED set of three groups: the top three at 100% (plus the ground truth
            # if it is not already there). Probabilities come from the full res["probs"]
            # vector: taking each cut-off's own top-k would give a different set of columns
            # every time, leaving the chart broken and meaningless.
            full = run_analysis(d["text"], doc_id, d["group"], 1.0, tag, None)
            watch = [t["group"] for t in full["topk"][:3]]
            if d["group"] not in watch:
                watch.append(d["group"])
            rows = []
            for k in (0.2, 0.4, 0.6, 0.8, 1.0):
                r = run_analysis(d["text"], doc_id, d["group"], k, tag, None)
                by_group = dict(zip(groups_all, r["probs"]))
                # The index must be NUMERIC so the x-axis sorts correctly. With string labels
                # "20%".."100%", Streamlit sorts alphabetically and 100% jumps to the front.
                rows.append({"Timeline retained (%)": int(k * 100),
                             **{g: float(by_group.get(g, 0.0)) for g in watch}})
        st.line_chart(pd.DataFrame(rows).set_index("Timeline retained (%)"), height=300)
        st.caption(f"The three leading groups at the 100% cut-off"
                   f"{', plus the ground truth ' + d['group'] if d['group'] not in watch[:3] else ''}. "
                   "The ground-truth curve rising as the timeline fills in is a good sign.")

    H(note("<b>How to read this.</b> The paper fits a straight line <b>m = A·t + B</b> through "
           "the five cut-offs (Eq. 27), where t is the fraction of the timeline retained. "
           "<b>Large A</b> means the model exploits new information well as the timeline fills "
           "in. <b>Large B</b> means it is already accurate in the early stage. Higher is "
           "better for both.<br><br>A <b>perfectly flat</b> line is NOT good robustness — it "
           "is a sign the model predicts an almost constant output and ignores its input."))


# ================================================================= PAGE 3
def page_group_profile():
    H(page_head(
        "Threat group behavioural profile",
        "The characteristic TTPs of each group, aggregated from contribution scores across the "
        "whole test set (corresponding to Fig. 5), then compared against the MITRE ATT&CK "
        "Groups v14 pages (Eq. 28-29)."))
    fp = OUTPUTS / "fig5_group_ttp_contribution.csv"
    if not fp.exists():
        need("<b>outputs/fig5_group_ttp_contribution.csv</b> is missing. Run Stage 6:<br>"
             "<code>python -m src.explain group-profile</code>")
    df = pd.read_csv(fp).set_index("group")
    grp = st.selectbox("Threat group", sorted(df.index))
    row = df.loc[grp].sort_values(ascending=False)
    top = row[row > 0].head(20)

    m = OUTPUTS / "tableXI_attck_match.csv"
    r = None
    if m.exists():
        t = pd.read_csv(m)
        sel = t[t["group"] == grp]
        r = sel.iloc[0] if len(sel) else None

    if r is not None and str(r["jaccard"]) not in ("", "nan"):
        H(kpi_row([("Matching ATT&CK group", str(r["attck_group"])),
                   ("Jaccard", f'{float(r["jaccard"]):.3f}', "Eq. 28"),
                   ("F1", f'{float(r["f1"]):.3f}', "Eq. 29"),
                   ("Found by this system", f'{int(r["n_ours"])} techniques'),
                   ("Listed by ATT&CK", f'{int(r["n_attck"])} techniques')]))
    elif r is not None:
        H(note(f"Group <b>{grp}</b> has no corresponding intrusion-set in ATT&CK v14, so no "
               "comparison is possible. This should be stated explicitly when reporting results."))
    else:
        H(note("The comparison table is missing. Run "
               "<code>python -m src.explain attck-match</code>"))

    c1, c2 = st.columns([3, 2])
    with c1:
        H(section(f"Highest-contributing TTPs — {grp}"))
        st.bar_chart(top, height=390)
        st.caption("Mean contribution score of the Technique nodes across documents "
                   "belonging to this group.")
    with c2:
        ref = ROOT / "data" / "reference" / "ttp_group_contribution_authors.csv"
        H(section("Comparison with the original authors' data"))
        if ref.exists():
            a = pd.read_csv(ref)
            if grp in a.columns:
                st.dataframe(a[["ttp", grp]].sort_values(grp, ascending=False).head(12),
                             hide_index=True, height=390, **STRETCH)
            else:
                H(note(f"The authors' table has no column for group <b>{grp}</b>."))
        else:
            H(note("<b>data/reference/ttp_group_contribution_authors.csv</b> is missing."))

    with st.expander("Show the full TTP × group matrix"):
        st.dataframe(df, **STRETCH)


# ================================================================= PAGE 4
def page_summary():
    H(page_head(
        "Consolidated results",
        "Displays the files already computed in Stages 5-6 under outputs/ — nothing is "
        "recomputed here."))
    any_file = False

    acc = acceptance_main()
    if acc:
        r, ok = acc
        H(verdict("Stage 5 acceptance", "PASS" if ok else "FAIL",
                  badge("Required: macro-F1 > 0.35 and Top-3 > 0.60", "green" if ok else "red")))
        st.write("")
        H(kpi_row([("macro-F1", f'{float(r["f1_macro"]):.4f}', "threshold 0.35"),
                   ("micro-F1", f'{float(r["f1_micro"]):.4f}'),
                   ("Top-1", f'{float(r["top1_micro"]):.4f}'),
                   ("Top-3", f'{float(r["top3_micro"]):.4f}', "threshold 0.60"),
                   ("Top-5", f'{float(r["top5_micro"]):.4f}'),
                   ("Classes predicted", f'{int(float(r.get("n_pred_cls", 0)))}')]))

    m = OUTPUTS / "metrics_all.csv"
    if m.exists():
        any_file = True
        H(section("Headline metrics and ablations (corresponding to Table VII)"))
        st.dataframe(pd.read_csv(m), hide_index=True, **STRETCH)
        st.caption("The full model compared against three ablations. An ablation beating the "
                   "full model is a sign of misconfiguration, not a scientific finding.")

    robs = sorted(OUTPUTS.glob("robustness_*.csv"))
    if robs:
        any_file = True
        H(section("Robustness over the timeline (Table VIII, Eq. 27)"))
        cols = st.columns(min(2, len(robs)))
        for i, fp in enumerate(robs):
            d = pd.read_csv(fp)
            flat = d["top1_micro"].nunique() == 1
            with cols[i % len(cols)]:
                H(f'<b>{fp.stem.replace("robustness_", "")}</b> &nbsp; '
                  + badge(f'A = {d["fit_A"].iloc[0]} · B = {d["fit_B"].iloc[0]}',
                          "red" if flat else "cyan"))
                dd = d.copy()
                dd["Retained (%)"] = (dd["keep_frac"] * 100).astype(int)
                st.line_chart(dd.set_index("Retained (%)")[["top1_micro", "top3_micro"]], height=210)
                if flat:
                    H(note("Perfectly flat line — <b>fake robustness</b>, do not report."))
        st.caption("The paper (Table VIII) reports A = 37.54 and B = 30.32 for micro Top-1. "
                   "Large A = exploits new information well; large B = accurate from the "
                   "earliest stage.")

    vals = sorted(OUTPUTS.glob("validity_*.csv"))
    if vals:
        any_file = True
        H(section("Feature representation quality (Table X, Eq. 24-26)"))
        st.dataframe(pd.concat([pd.read_csv(f).assign(model=f.stem.replace("validity_", ""))
                                for f in vals]), hide_index=True, **STRETCH)
        H(note("The first three columns are measured on <b>V_G</b>, the graph representation "
               "fed to the classifier, exactly as in the paper. The three <code>_z</code> "
               "columns are measured on the projection head output; because InfoNCE "
               "L2-normalises, those vectors retain only <b>angular</b> information, so "
               "Euclidean distances on them are badly skewed. Report only the first three "
               "columns. The paper reports D_intra 10.862 · D_inter 33.436 · "
               "<b>D_separ 3.078</b>."))

    t = OUTPUTS / "tableXI_attck_match.csv"
    if t.exists():
        any_file = True
        H(section("Comparison with MITRE ATT&CK (corresponding to Table XI)"))
        d = pd.read_csv(t)
        jv = pd.to_numeric(d["jaccard"], errors="coerce")
        H(kpi_row([("Groups matched", f'{int(jv.notna().sum())}/{len(d)}'),
                   ("Mean Jaccard", f'{jv.mean():.4f}', "paper: 0.2991"),
                   ("Highest Jaccard", f'{jv.max():.4f}',
                    str(d.loc[jv.idxmax(), "group"]) if jv.notna().any() else "")]))
        st.dataframe(d, hide_index=True, **STRETCH)

    imgs = list(OUTPUTS.glob("tsne_*.png")) + list(OUTPUTS.glob("fig5_heatmap.png")) \
        + list(OUTPUTS.glob("cskg_*.png"))
    if imgs:
        any_file = True
        H(section("Figures"))
        for i in range(0, len(imgs), 2):
            cols = st.columns(2)
            for c, p in zip(cols, imgs[i:i + 2]):
                c.image(str(p), caption=p.name, **STRETCH)

    hs = sorted(OUTPUTS.glob("history_*.csv"))
    if hs:
        any_file = True
        H(section("Learning curves per epoch"))
        with st.expander("Show the learning curves of all four configurations"):
            for fp in hs:
                st.markdown(f"**{fp.stem.replace('history_', '')}**")
                dh = pd.read_csv(fp).set_index("epoch")
                keep = [c for c in ("train_loss", "train_bce", "train_cl", "val_macro",
                                    "val_bal_acc", "val_micro") if c in dh.columns]
                st.line_chart(dh[keep] if keep else dh, height=210)
            st.caption("train_cl column: the chance level of InfoNCE is ln(K+1) = 1.6094 for "
                       "K = 4. A curve sitting at that value means contrastive learning is "
                       "not learning anything.")

    if not any_file:
        need("<b>outputs/</b> contains no results yet. Run Stages 5-6 first:<br>"
             "<code>bash scripts/train_all.sh</code><br>"
             "<code>python -m src.explain group-profile</code><br>"
             "<code>python -m src.explain attck-match</code>")


{PAGES[0]: page_analyze, PAGES[1]: page_robustness,
 PAGES[2]: page_group_profile, PAGES[3]: page_summary}[page]()
