"""Shared UI components for the Streamlit app (dark SOC theme).

Contains only CSS and pure HTML-producing functions, with no dependency on the model
logic, so it can be tested in isolation. Every data-derived string is passed through
html.escape before being inserted into the markup.
"""
from html import escape

# --------------------------------------------------------------------- palette
PALETTE = {
    "bg": "#0b0f14", "panel": "#111922", "panel2": "#0d141c", "bd": "#1e2a38",
    "fg": "#e6edf3", "muted": "#8b9aad",
    "cyan": "#22d3ee", "green": "#34d399", "amber": "#fbbf24",
    "red": "#f87171", "violet": "#a78bfa", "blue": "#60a5fa",
}

# Badge colour per node type in the 19-type ontology (Fig. 3).
NODE_TONE = {
    "State": "cyan", "Technique": "violet", "Tactic": "violet",
    "IP": "amber", "Domain": "amber", "URL": "amber", "Email": "amber",
    "Hash": "red", "CVE": "red", "CMD": "red", "API": "red", "Registry": "red",
    "Filename": "blue", "Port": "blue", "Protocol": "blue", "Account": "blue", "MAC": "blue",
    "Time": "green", "Geo-location": "green",
}

CSS = """
<style>
:root{
  --bg:#0b0f14; --panel:#111922; --panel2:#0d141c; --bd:#1e2a38;
  --fg:#e6edf3; --muted:#8b9aad;
  --cyan:#22d3ee; --green:#34d399; --amber:#fbbf24; --red:#f87171;
  --violet:#a78bfa; --blue:#60a5fa;
}
.stApp{background:var(--bg);}
#MainMenu, footer{visibility:hidden;}
.block-container{padding-top:1.6rem; padding-bottom:3rem; max-width:1500px;}
section[data-testid="stSidebar"]{background:var(--panel2); border-right:1px solid var(--bd);}
section[data-testid="stSidebar"] .block-container{padding-top:1.2rem;}

/* ---------- sidebar branding ---------- */
.brand{display:flex; align-items:center; gap:.65rem; padding:.2rem 0 1rem;}
.brand-mark{width:34px; height:34px; border-radius:9px; flex:0 0 34px;
  background:linear-gradient(135deg,var(--cyan),var(--violet));
  display:flex; align-items:center; justify-content:center;
  font-size:17px; box-shadow:0 0 16px rgba(34,211,238,.30);}
.brand-name{font-size:1.02rem; font-weight:700; letter-spacing:.4px; line-height:1.15;}
.brand-sub{font-size:.68rem; color:var(--muted); letter-spacing:.7px; text-transform:uppercase;}

/* ---------- page header ---------- */
.page-head{border-bottom:1px solid var(--bd); padding-bottom:.85rem; margin-bottom:1.3rem;}
.page-title{font-size:1.5rem; font-weight:700; letter-spacing:-.2px; margin:0;}
.page-sub{color:var(--muted); font-size:.86rem; margin-top:.3rem; line-height:1.5;}

/* ---------- section heading ---------- */
.sec{font-size:.72rem; font-weight:700; letter-spacing:1.1px; text-transform:uppercase;
  color:var(--muted); margin:1.5rem 0 .6rem; display:flex; align-items:center; gap:.6rem;}
.sec::after{content:""; flex:1; height:1px; background:var(--bd);}

/* ---------- KPI cards ---------- */
.kpi-row{display:flex; gap:.7rem; flex-wrap:wrap;}
.kpi{flex:1 1 0; min-width:118px; background:var(--panel); border:1px solid var(--bd);
  border-radius:11px; padding:.75rem .85rem;}
.kpi-l{font-size:.66rem; color:var(--muted); text-transform:uppercase; letter-spacing:.8px;}
.kpi-v{font-size:1.42rem; font-weight:700; margin-top:.18rem; line-height:1.15;
  font-variant-numeric:tabular-nums;}
.kpi-s{font-size:.71rem; color:var(--muted); margin-top:.12rem;}

/* ---------- verdict block ---------- */
.verdict{background:linear-gradient(135deg,rgba(34,211,238,.09),rgba(167,139,250,.06));
  border:1px solid var(--bd); border-left:3px solid var(--cyan);
  border-radius:12px; padding:1rem 1.15rem; display:flex; align-items:center;
  justify-content:space-between; gap:1rem; flex-wrap:wrap;}
.verdict-l{font-size:.68rem; color:var(--muted); text-transform:uppercase; letter-spacing:1px;}
.verdict-v{font-size:1.95rem; font-weight:700; line-height:1.15; margin-top:.1rem;}
.verdict-r{text-align:right;}

/* ---------- badges ---------- */
.badge{display:inline-block; padding:.13rem .5rem; border-radius:5px; font-size:.7rem;
  font-weight:600; letter-spacing:.3px; white-space:nowrap;}
.b-cyan{background:rgba(34,211,238,.13); color:var(--cyan); border:1px solid rgba(34,211,238,.3);}
.b-green{background:rgba(52,211,153,.13); color:var(--green); border:1px solid rgba(52,211,153,.3);}
.b-amber{background:rgba(251,191,36,.13); color:var(--amber); border:1px solid rgba(251,191,36,.3);}
.b-red{background:rgba(248,113,113,.13); color:var(--red); border:1px solid rgba(248,113,113,.3);}
.b-violet{background:rgba(167,139,250,.13); color:var(--violet); border:1px solid rgba(167,139,250,.3);}
.b-blue{background:rgba(96,165,250,.13); color:var(--blue); border:1px solid rgba(96,165,250,.3);}
.b-muted{background:rgba(139,154,173,.11); color:var(--muted); border:1px solid rgba(139,154,173,.25);}

/* ---------- top-k ranking list ---------- */
.rank{background:var(--panel); border:1px solid var(--bd); border-radius:11px; padding:.75rem .9rem;}
.rk{display:flex; align-items:center; gap:.7rem; padding:.36rem 0;}
.rk-i{width:19px; height:19px; flex:0 0 19px; border-radius:5px; background:rgba(139,154,173,.13);
  color:var(--muted); font-size:.68rem; font-weight:700; display:flex;
  align-items:center; justify-content:center;}
.rk-1 .rk-i{background:rgba(34,211,238,.18); color:var(--cyan);}
.rk-n{flex:0 0 132px; font-size:.85rem; font-weight:600; overflow:hidden;
  text-overflow:ellipsis; white-space:nowrap;}
.rk-b{flex:1; height:7px; background:rgba(139,154,173,.11); border-radius:4px; overflow:hidden;}
.rk-b>div{height:100%; border-radius:4px; background:linear-gradient(90deg,var(--cyan),var(--violet));}
.rk-1 .rk-b>div{box-shadow:0 0 9px rgba(34,211,238,.5);}
.rk-v{flex:0 0 54px; text-align:right; font-size:.79rem; color:var(--muted);
  font-variant-numeric:tabular-nums;}

/* ---------- pipeline progress ---------- */
.steps{display:flex; gap:.55rem; flex-wrap:wrap;}
.step{flex:1 1 0; min-width:150px; background:var(--panel); border:1px solid var(--bd);
  border-radius:10px; padding:.6rem .75rem; position:relative;}
.step::before{content:""; position:absolute; left:0; top:12%; height:76%; width:2px;
  border-radius:2px; background:var(--green);}
.step-n{font-size:.73rem; font-weight:700; display:flex; align-items:center; gap:.35rem;}
.step-i{font-size:.71rem; color:var(--muted); margin-top:.22rem; line-height:1.4;}

/* ---------- sidebar system status ---------- */
.stat{display:flex; align-items:center; justify-content:space-between; gap:.5rem;
  padding:.24rem 0; font-size:.76rem; border-bottom:1px dashed rgba(30,42,56,.85);}
.stat:last-child{border-bottom:none;}
.stat-k{color:var(--muted);}
.dot{display:inline-block; width:7px; height:7px; border-radius:50%; margin-right:.35rem;}
.d-ok{background:var(--green); box-shadow:0 0 6px var(--green);}
.d-no{background:var(--red);}
.d-wa{background:var(--amber);}

/* ---------- notes ---------- */
.note{background:var(--panel); border:1px solid var(--bd); border-left:3px solid var(--amber);
  border-radius:9px; padding:.7rem .9rem; font-size:.79rem; color:var(--muted); line-height:1.6;}
.note code{color:var(--cyan); background:rgba(34,211,238,.09); padding:.05rem .3rem;
  border-radius:4px; font-size:.94em;}
.note b{color:var(--fg);}

/* ---------- tweaks to built-in widgets ---------- */
div[data-testid="stDataFrame"]{border:1px solid var(--bd); border-radius:10px;}
.stTabs [data-baseweb="tab-list"]{gap:.25rem; border-bottom:1px solid var(--bd);}
.stTabs [data-baseweb="tab"]{height:36px; font-size:.84rem;}
div[data-testid="stExpander"] details{border:1px solid var(--bd); border-radius:10px;
  background:var(--panel);}
section[data-testid="stSidebar"] div[role="radiogroup"] label{padding:.24rem 0; font-size:.87rem;}
.stSlider [data-baseweb="slider"]{padding-top:.4rem;}
</style>
"""


# --------------------------------------------------------------------- components
def badge(text: str, tone: str = "muted") -> str:
    return f'<span class="badge b-{tone}">{escape(str(text))}</span>'


def node_badge(node_type: str) -> str:
    return badge(node_type, NODE_TONE.get(node_type, "muted"))


def page_head(title: str, sub: str = "") -> str:
    s = f'<div class="page-sub">{sub}</div>' if sub else ""
    return f'<div class="page-head"><div class="page-title">{escape(title)}</div>{s}</div>'


def section(title: str) -> str:
    return f'<div class="sec">{escape(title)}</div>'


def kpi_row(items: list[tuple]) -> str:
    """items: a list of (label, value) or (label, value, caption) tuples."""
    cells = []
    for it in items:
        label, value = it[0], it[1]
        sub = it[2] if len(it) > 2 else ""
        sub_html = f'<div class="kpi-s">{escape(str(sub))}</div>' if sub else ""
        cells.append(f'<div class="kpi"><div class="kpi-l">{escape(str(label))}</div>'
                     f'<div class="kpi-v">{escape(str(value))}</div>{sub_html}</div>')
    return f'<div class="kpi-row">{"".join(cells)}</div>'


def verdict(label: str, value: str, right_html: str = "") -> str:
    return (f'<div class="verdict"><div><div class="verdict-l">{escape(label)}</div>'
            f'<div class="verdict-v">{escape(value)}</div></div>'
            f'<div class="verdict-r">{right_html}</div></div>')


def rank_list(rows: list[dict], key_name: str = "group", key_val: str = "prob") -> str:
    """Ranking list with bars whose width is proportional to the probability."""
    if not rows:
        return ""
    vmax = max(float(r[key_val]) for r in rows) or 1.0
    out = []
    for i, r in enumerate(rows, 1):
        v = float(r[key_val])
        w = max(2.0, 100.0 * v / vmax)
        out.append(f'<div class="rk rk-{i}"><div class="rk-i">{i}</div>'
                   f'<div class="rk-n">{escape(str(r[key_name]))}</div>'
                   f'<div class="rk-b"><div style="width:{w:.1f}%"></div></div>'
                   f'<div class="rk-v">{v:.3f}</div></div>')
    return f'<div class="rank">{"".join(out)}</div>'


# Step keys are produced by src/inference.py; these are the display labels.
STEP_LABEL = {
    "IOC": ("◈", "1 · Extract IOCs"),
    "TTP": ("△", "2 · Extract TTPs"),
    "CSKG": ("⬡", "3 · Build CSKG"),
    "Predict": ("◉", "4 · Attribute group"),
}


def steps_bar(steps: list) -> str:
    cells = []
    for name, info in steps:
        ic, label = STEP_LABEL.get(name, ("●", str(name)))
        cells.append(f'<div class="step"><div class="step-n">{ic} {escape(label)}</div>'
                     f'<div class="step-i">{escape(str(info))}</div></div>')
    return f'<div class="steps">{"".join(cells)}</div>'


def stat_line(key: str, value: str, state: str = "ok") -> str:
    """state: ok | no | wa"""
    return (f'<div class="stat"><span class="stat-k">{escape(key)}</span>'
            f'<span><span class="dot d-{state}"></span>{escape(value)}</span></div>')


def note(html_body: str) -> str:
    return f'<div class="note">{html_body}</div>'
