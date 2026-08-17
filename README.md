# ThreatMamba — an open reproduction of ThreatMAMBA

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.8%2B-ee4c2c.svg)](https://pytorch.org/)

A from-scratch reproduction of **ThreatMAMBA** (Ge et al., *IEEE TIFS*, 2026 —
[10.1109/TIFS.2026.3685967](https://doi.org/10.1109/TIFS.2026.3685967)), an end-to-end
pipeline for **cyber threat attribution**: read a Cyber Threat Intelligence report, extract
its IOCs and TTPs, assemble a time-ordered Cybersecurity Knowledge Graph, and attribute the
report to a threat group — with per-node explanations grounded in MITRE ATT&CK.

Every stage is implemented and runnable. This repository also documents, in the code
itself, the configuration mistakes that broke the first run and how they were diagnosed —
those notes are the practical part of reproducing this paper.

> 🇻🇳 Bản tiếng Việt: [README.vi.md](README.vi.md)

---

## Table of contents

- [What this does](#what-this-does)
- [Pipeline](#pipeline)
- [Results](#results)
- [Installation](#installation)
- [Running the pipeline](#running-the-pipeline)
- [Demo UI](#demo-ui)
- [Interpreting the output](#interpreting-the-output)
- [Repository layout](#repository-layout)
- [Testing](#testing)
- [Troubleshooting](#troubleshooting)
- [Scope and deviations from the paper](#scope-and-deviations-from-the-paper)
- [Citation](#citation)
- [Acknowledgements and licence](#acknowledgements-and-licence)

---

## What this does

Given an English-language CTI report as plain text, the system returns:

- a ranked list of candidate **threat groups** with calibrated per-class probabilities;
- the **IOCs** found in the text (13 types) plus any discovered by live threat-intel lookups
  (Time, Geo-location, CMD, API);
- the **TTPs** (MITRE ATT&CK tactics and techniques) predicted per sentence;
- a **Cybersecurity Knowledge Graph (CSKG)** over 20 node types and 11 relation types,
  ordered along the report's timeline;
- a **contribution score for every node**, so an analyst can see which piece of evidence
  drove the attribution;
- a **robustness curve** showing how the prediction holds up when only the first 20–100% of
  the timeline is available — that is, early in an investigation.

---

## Pipeline

```
CTI report (text)
   │
   ├─ Stage 1  data preparation ....... clean, filter groups (>=30 docs), 8:2 split
   │
   ├─ Stage 2  IOC extraction ......... ioc-finder + regexes -> 13 IOC types
   │           IOCHunter daemon ....... UCT method selection (Eq. 1) -> VirusTotal /
   │                                    AlienVault OTX / RapidDNS -> LLM screening
   │                                    -> Time, Geo-location, CMD, API nodes
   │
   ├─ Stage 3  TTP extraction ......... SecureBERT (frozen) -> projection -> SSM
   │                                    -> per-label Gaussian attention (Eq. 2-5)
   │
   ├─ Stage 4  CSKG construction ...... 20 node types, 11 relations, timeline backbone (Eq. 6)
   │
   ├─ Stage 5  classifier ............. relational GAT (Eq. 9-11) ‖ MAMBA over the state
   │                                    sequence (Eq. 12-14) -> AvgPool‖MaxPool (Eq. 16)
   │                                    -> sigmoid MLP (Eq. 17)
   │                                    loss = BCE + λ·InfoNCE (Eq. 18-20)
   │
   ├─ Stage 6  evaluation & XAI ....... F1, Top-k, robustness (Eq. 27), representation
   │                                    validity (Eq. 24-26), ATT&CK matching (Eq. 28-29)
   │
   └─ Stage 7  demo UI ................ four-page Streamlit application
```

### Architecture notes

**Relational GAT.** Each of the 11 relation types has its own attention vector (Eq. 10);
3 layers, 4 heads, dimension 48, residual connections plus a jump link. Implemented with
plain PyTorch scatter operations, so `torch_geometric` is not a dependency.

**MAMBA branch.** A 3-layer selective state space model over the sequence of State nodes,
which is what encodes the temporal ordering of the attack. The official `mamba-ssm` CUDA
kernel is used when available; a pure-PyTorch `SimpleSSM` fallback (causal conv + selective
gating + linear scan) is provided for machines without a suitable GPU.

**Contrastive learning.** Positive views apply Gaussian feature noise and Bernoulli edge
dropping while preserving the temporal backbone (Eq. 7). Negative views splice the timelines
of two differently-labelled reports with an offset φ ∈ (−100, 100)\\{0} and rewire the ATT&CK
kill-chain relations over the merged graph (Eq. 8).

---

## Results

Measured on the reproduction described here. The dataset is the CTI corpus published with
the paper: 10,510 rows → 3,589 documents of sufficient length → **20 groups / 3,144
documents** after keeping groups with at least 30 documents (2,515 train / 629 test).

| Component | Metric | Value |
|---|---|---|
| Stage 1 — ATT&CK v14 parse | techniques / tactics / groups | 625 / 14 / 142 |
| Stage 3 — TTP module (D1) | 12,330 samples, 14 tactics, 540 technique columns (333 with ≥5 positives) | — |
| Stage 4 — CSKG | mean nodes / edges per graph | ~37 / ~96 (paper ~53 / ~147) |
| Stage 4 — CSKG without TTP nodes | mean nodes / edges | ~13 / ~18 |
| Stage 6 — ATT&CK group mapping | groups matched to an intrusion-set | 15 / 20 |

The Stage 5 acceptance criteria used here are **macro-F1 > 0.35** and **Top-3 > 0.60**;
`python -m src.evaluate all` prints PASS/FAIL against them and writes
`outputs/metrics_all.csv`.

Five of the twenty dataset groups (Nitro, SEA, Scarab, TwoForOne) have no corresponding
intrusion-set in ATT&CK v14 and are reported as unmatched rather than silently scored. The
verified aliases are: Energetic Bear → Dragonfly, Hidden Cobra → Lazarus Group, Quedagh →
Sandworm Team, Tick → BRONZE BUTLER, Waterbug → Turla, MageCart → FIN6.

> **Reproducing exact numbers.** Stage 4 **must** run after Stage 3. Building the CSKG
> without a trained TTP module yields ~13 nodes per graph instead of ~37, and every
> downstream number changes accordingly.

---

## Installation

**Reference environment:** Windows 11 + WSL2 (Ubuntu 22.04/24.04) with an NVIDIA GPU
(developed against an RTX 5060 Ti 16 GB, Blackwell `sm_120`), ~25 GB free disk. Native Linux
works too. Free accounts are needed for VirusTotal and AlienVault OTX; RapidDNS needs none.

```bash
git clone https://github.com/<your-username>/threatmamba-repro
cd threatmamba-repro
bash scripts/setup_env.sh
```

`setup_env.sh` creates a `.venv` on **Python 3.11**, installs **PyTorch 2.8.0 + cu128**
(which supports Blackwell `sm_120`), and all libraries for Stages 1–7.

Three things worth knowing about this step:

**Python must be 3.10–3.13, ideally 3.11.** Python 3.14 has no wheels for `mamba-ssm`,
`causal-conv1d` or several ML libraries — this is the single most common installation
failure.

**`mamba-ssm` is deliberately not installed.** The prebuilt wheels only compile up to
`sm_100`, so on a `sm_120` GPU they fail with *no kernel image is available*. Getting the
real kernel requires building from source (see below).

**The project runs Stages 1–7 without `mamba-ssm`** using the fallback:

```bash
export THREATMAMBA_SSM=simple      # consider adding this to ~/.bashrc
```

The fallback is a pure-PyTorch SSM: slower to train, but computationally faithful.

### Optional — build the real mamba-ssm kernel

```bash
bash scripts/setup_mamba.sh              # builds for sm_120 only; fastest
MAX_JOBS=2 bash scripts/setup_mamba.sh   # if the build is killed for lack of RAM
```

The script handles the three things a manual build usually trips over: it installs **CUDA
Toolkit 12.8** if the current `nvcc` is older (apt's 12.4 cannot emit `sm_120` code),
installs **g++-14/13** if the system `g++` is newer than 14 (CUDA 12.8 rejects g++ 15), and
patches `setup.py` to build only `sm_120` instead of a dozen architectures. Expect 10–40
minutes.

### Ollama and API keys

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull qwen3:8b        # ~5 GB

cp .env.example .env && $EDITOR .env
```

`VT_API_KEY`: virustotal.com → sign up → click your avatar → **API key**.
`OTX_API_KEY`: otx.alienvault.com → sign up → **Settings** → OTX Key.

If Ollama is already installed on Windows there is no need to reinstall it inside WSL —
point `OLLAMA_HOST` at the Windows host instead.

### Verify the environment

```bash
source .venv/bin/activate
bash scripts/smoke_test.sh
```

This passes when all 11 checks report PASS with no FAIL and no SKIP: `env`, `gpu-driver`,
`python`, `toolchain`, `cuda`, `mamba`, `ollama`, `securebert`, `vt`, `otx`, `rapiddns`.

The `mamba` check passes with either the real kernel or the fallback, and prints which mode
is active; it fails only when both are broken. The first run downloads SecureBERT (~500 MB)
and loads Qwen into VRAM, which takes a minute or two. Individual checks can be run alone:

```bash
python scripts/checks/check_cuda.py
python scripts/checks/check_apis.py vt
```

---

## Running the pipeline

Every stage can be run independently. `bash scripts/run_all.sh` runs them in order, but
Stage 5 takes hours, so working through them once by hand is recommended.

### Stage 1 — data

```bash
bash scripts/download_data.sh       # MuscleFish CSVs + enterprise-attack-14.1.json
python -m src.data_prep all         # prepare -> prepare-d1 -> attck -> demo-subset -> stats
```

Of the ten files in the upstream data repository only three are actually needed:
`CTI2Attacker.csv` (D2, for the classifier), `CTI2TTPs.csv` (D1, for the TTP module) and
`TTP_Group_Contribution.csv` (the authors' contribution table, used to cross-check Fig. 5).
The `.xlsx` files duplicate the CSVs.

*Acceptance:* a sensible statistics table plus a line reading `[ATT&CK v14] … -> PASS`.

### Stage 2 — IOC enrichment daemon

```bash
bash scripts/run_enrichment_daemon.sh          # start (background, resumable)
bash scripts/run_enrichment_daemon.sh status   # progress (= acceptance check)
bash scripts/run_enrichment_daemon.sh stop
```

The daemon reads `data/demo_subset.txt` and runs Algorithm 1: rank hunting methods by UCT
(Eq. 1) → call VirusTotal / OTX / RapidDNS → have Qwen3-8B score the candidates using the
Fig. 2 prompt (JSON validated, 3 retries) → accept candidates scoring ≥ 6. It respects the
VirusTotal rate limit (15 s/request, 480 requests/day), checkpoints after **every IOC**, and
logs to `outputs/enrich.log`. Once the demo subset is exhausted it continues into the
training split.

*Acceptance (after ~24 h):* `python scripts/check_enrichment.py` reports ≥ 5 documents
carrying Time/Geo-location nodes and no unrecoverable crash.

> **Hard rule:** enriched data is **never** mixed into the classifier's training set. It
> exists for the demo UI and to illustrate the full 19-type ontology.

Start the daemon early and run Stages 3–6 alongside it.

### Stage 3 — TTP extraction

```bash
python -m src.ttp_extract encode     # precompute SecureBERT embeddings for D1 (cached)
python -m src.ttp_extract train      # BCE with pos_weight, early stopping on macro AP
python -m src.ttp_extract eval       # ACCEPTANCE: tactics micro-F1 >= 0.70
```

Try it directly: `python -m src.ttp_extract extract --text "APT29 used spearphishing..."`.

### Stage 4 — CSKG

```bash
python -m src.cskg_builder build --split all --mode train      # used by the classifier
python -m src.cskg_builder stats                               # ACCEPTANCE
python -m src.cskg_builder build --split all --mode enriched   # for enriched documents
python -m src.cskg_builder render --doc $(head -1 data/demo_subset.txt)
```

*Acceptance:* nodes and edges per graph in the same order of magnitude as the paper.

### Stage 5 — classifier

The recommended path trains all four configurations with one config, one seed and one
stopping rule, then evaluates them:

```bash
bash scripts/train_all.sh
```

Or individually:

```bash
python -m src.train                  # main
python -m src.train --no-mamba       # ablation w/o MAMBA
python -m src.train --no-cl          # ablation w/o contrastive learning
python -m src.train --no-iochunter   # ablation w/o IOCHunter (drops co-occur/hunting edges)
```

Progress: `outputs/history_{tag}.csv`. Checkpoints: `outputs/model_{tag}.pt`.

#### Reading the training log

```
[main] ep  12 | lam 0.1  | loss 0.3021 (bce 0.2870 cl 1.5090) | tr bal 0.6120 | val micro 0.5410 macro 0.3520 bal 0.3710 | pred 18/20  *
```

| Column | Meaning | Warning sign |
|---|---|---|
| `lam` | the λ currently applied (0 during warm-up) | — |
| `bce` / `cl` | the two loss components, separately | `λ·cl` exceeding `bce` → CL is swamping BCE |
| `cl` | InfoNCE. Chance level = ln(K+1) = **1.6094** for K=4 | stuck near 1.61 → CL is learning nothing |
| `pred` | how many classes the model **actually** predicts | ≤ 5/20 → near-degenerate |
| `*` | best epoch so far; checkpoint written | — |

#### Three bugs worth knowing about

The first run produced a paradox: the `no_cl` ablation reached macro-F1 0.3515 (**passing**)
while the main model reached 0.0389 (**failing**). The cause was configuration, not method:

1. **`cl_lambda` = 1.0 let contrastive learning swamp BCE.** With K=4 negatives, InfoNCE
   starts around ln(5) = 1.61 while BCE starts around 0.35 — more than 80% of the gradient
   came from the contrastive term, so the model never learned to classify. New default:
   **0.1**.
2. **Early stopping on validation macro-F1 was too noisy.** 20 classes over ~250 validation
   samples, many classes with 2–4 samples: `main` hit a lucky record at epoch 2 and stopped
   at epoch 7 while `no_cl` ran 41 epochs, making the ablation table meaningless. Selection
   now uses **balanced accuracy** (macro recall) — it still weights rare classes like
   macro-F1 but is smoother, because no precision term collapses to 0 when a class is never
   predicted. Plus `patience: 10` and `min_epochs: 20`.
3. **Contrastive learning had no warm-up.** `cl_warmup_epochs: 5` now runs BCE alone first.

Corroborating evidence: `D_separ` for `main` was 0.719 (< 1, clusters overlapping) against
1.734 for `no_cl`. Contrastive learning exists to *increase* separation; enabling it making
things 2.4× worse is a reliable signal that λ is the problem.

#### λ grid search

```bash
bash scripts/grid_cl.sh                    # tries {0.1, 0.5, 1.0}
LAMS="0.05 0.1 0.2" bash scripts/grid_cl.sh
bash scripts/grid_cl.sh --epochs 40        # faster pass
```

Results are collected in `outputs/grid_cl_summary.csv`. Write the chosen λ into
`configs/default.yaml` (`train.cl_lambda`) and rerun `bash scripts/train_all.sh` so all four
ablations share it.

### Stage 6 — evaluation and explanations

```bash
python -m src.evaluate all --robustness --validity
python -m src.explain group-profile     # Fig. 5 style heatmap
python -m src.explain attck-match       # Table XI style comparison
python -m src.explain doc --doc d00042  # per-node contributions for one document
```

Writes to `outputs/`: `metrics_all.csv` (micro/macro F1, Top-1/3/5), `robustness_{tag}.csv`
(5 timeline cut-offs plus the A/B coefficients of Eq. 27), `validity_{tag}.csv`
(D_intra/D_inter/D_separ), `tsne_{tag}.png`, `fig5_heatmap.png` +
`fig5_group_ttp_contribution.csv`, `tableXI_attck_match.csv`.

---

## Demo UI

```bash
cd <repository root>          # REQUIRED
streamlit run app/streamlit_app.py
```

It must be launched from the repository root so Streamlit picks up `.streamlit/config.toml`
(the dark theme). Open the URL Streamlit prints, by default `http://localhost:8501`. From
WSL2 this opens fine in a Windows browser.

Three files control the interface: `.streamlit/config.toml` (colours for built-in widgets),
`app/ui_kit.py` (CSS and custom components), `app/streamlit_app.py` (the four pages).

The pie-node graph is also dark-themed, but **`app/pie_node_graph.html` is the original
authors' template, kept byte-for-byte** (including its Chinese comments — it is third-party
code and `scripts/download_data.sh` re-fetches it from upstream). Recolouring happens as
string substitution at render time (`_DARK_SWAP` in `src/inference.py`), so the template on
disk stays directly comparable with upstream.

The sidebar carries a **System status** panel: device (CUDA/CPU), Stage 1 document count,
whether the TTP module exists, checkpoint count, enriched document count, and **Stage 5
acceptance PASS/FAIL** read directly from `outputs/metrics_all.csv`.

**Page 1 — CTI report analysis.** Choose a document from the demo set (marked ✓ when
enriched) or paste any report. Shows the four-step pipeline progress, a Top-5 bar chart with
probabilities, the highest-contributing nodes (Eq. 21–23), the TTP and IOC tables (the
`Source` column says whether an IOC came from the text or from a specific hunting function),
and the **pie-node CSKG graph** using the authors' own ECharts template — State nodes are
pies of TTP contribution share, node size is importance, edge opacity is contribution
(as in Fig. 6c). A "Live enrichment" checkbox calls IOCHunter for real (consumes VirusTotal
quota; warned in advance).

**Page 2 — Robustness over time.** Drag the slider from 20% to 100% of the timeline; the
report is truncated, inference re-runs, and the Top-5 updates immediately. A line chart
tracks the top-3 probabilities across cut-offs. Each cut-off is cached, so the slider is
instant.

**Page 3 — Threat group profile.** Bar chart of a group's highest-contributing TTPs plus
Jaccard/F1 against the MITRE ATT&CK Groups page, side by side with the original authors'
data.

**Page 4 — Consolidated results.** Displays (never recomputes) everything in `outputs/`:
metrics and ablations, robustness, validity, t-SNE, heatmap, Table XI, learning curves.

Acceptance:

```bash
python tests/test_ui_smoke.py
```

This runs all four pages through `AppTest` and times them: **a fresh document in < 60 s**
(without live enrichment) and **each slider position in < 2 s**. A page that lacks data from
an earlier stage shows a warning and stops — that is correct behaviour, not a failure.

Picking documents for a live demo:

```bash
python scripts/pick_demo_docs.py -n 5
```

This scores the whole test split and selects documents that are correctly attributed, have a
wide top-1/top-2 margin, produce a rich graph, come from different groups, and are still
correct at the 60% timeline cut-off.

---

## Interpreting the output

Two automatic warnings deserve attention before any number is quoted.

**`!! FAKE ROBUSTNESS`.** A degenerate model predicts an almost constant output, so
truncating the timeline changes nothing: the Table VIII curve is perfectly flat and the
coefficient A ≈ 0, which *looks* like exceptional robustness. It is the most dangerous
misreading in this pipeline — a small A here means the model is **ignoring its input**, not
that it is robust. Observed example: `main` returned top-3 = 0.4499 identically at all five
cut-offs (20/40/60/80/100%), A = 0.0319, while `no_cl` climbed from 0.391 to 0.539 with
A = 0.1734 — steeper, but a *real* curve. When this warning fires, discard the table.

**`!! ABLATION … BEATS the main model`.** If removing a component improves results
substantially, that component is almost certainly switched on incorrectly (e.g. `cl_lambda`
far too high) rather than being useless. That is a configuration bug, not a scientific
finding — fix the configuration and rerun before concluding anything.

The `n_pred_cls` column in `metrics_all.csv` reports how many classes the model actually
predicts. A low value is the earliest sign of degeneration.

For the validity table, report only the first three columns: they are measured on **V_G**,
the graph representation fed to the classifier, exactly as in the paper. The `_z` columns
are measured on the contrastive projection head, which InfoNCE L2-normalises — those vectors
retain only angular information, so Euclidean distances on them are meaningless in this
context.

---

## Repository layout

```
threatmamba-repro/
├── README.md / README.vi.md / LICENSE / CITATION.cff
├── requirements.txt / .env.example
├── configs/default.yaml          # Table V hyper-parameters + Stage 1-7 settings
├── scripts/
│   ├── setup_env.sh              # Stage 0: environment
│   ├── setup_mamba.sh            # Stage 0: build mamba-ssm from source (optional)
│   ├── smoke_test.sh             # Stage 0: ACCEPTANCE (+ checks/)
│   ├── download_data.sh          # Stage 1: fetch data
│   ├── run_enrichment_daemon.sh  # Stage 2: start/stop/status
│   ├── check_enrichment.py       # Stage 2: ACCEPTANCE
│   ├── prune_enrich_state.py     # Stage 2: selectively clean daemon state
│   ├── train_all.sh              # Stage 5: 4 ablations, one config -> Stage 6
│   ├── grid_cl.sh                # Stage 5: λ grid search
│   ├── pick_demo_docs.py         # Stage 7: pick reliable demo documents
│   ├── patch_cuda_glibc.py       # fixes the CUDA/glibc >= 2.41 header conflict
│   ├── reset_env.sh              # clean reinstall
│   └── run_all.sh                # canonical Stage 1->7 sequence
├── src/
│   ├── utils.py, encoder.py, ssm.py                    # helpers, SecureBERT, SSM (+fallback)
│   ├── data_prep.py                                    # Stage 1
│   ├── ioc_extract.py, ioc_hunter/                     # Stage 2 (uct, llm_agent, apis, runner)
│   ├── ttp_extract.py                                  # Stage 3
│   ├── cskg_builder.py                                 # Stage 4
│   ├── contrastive.py, model.py, losses.py, train.py   # Stage 5
│   ├── evaluate.py, explain.py                         # Stage 6
│   └── inference.py                                    # Stage 7 (shared inference path)
├── .streamlit/config.toml        # dark SOC theme
├── app/
│   ├── streamlit_app.py          # four-page UI
│   ├── ui_kit.py                 # CSS + custom components
│   └── pie_node_graph.html       # the authors' pie-node template (unmodified)
├── tests/
│   ├── test_cpu_pipeline.py      # 13 checks, no GPU required
│   └── test_ui_smoke.py          # Stage 7 ACCEPTANCE
├── data/{raw,processed,cskg,enriched,attck,reference}/
└── outputs/                      # checkpoints, CSV tables, PNG figures, daemon log
```

---

## Testing

```bash
THREATMAMBA_FAKE_ENCODER=1 THREATMAMBA_SSM=simple python tests/test_cpu_pipeline.py
```

13 checks covering the entire tensor pipeline (IOC → CSKG train/enriched → timeline
truncation → contrastive views → all four model configurations → losses → metrics →
explanations → Gaussian attention → UCT → LLM agent). Useful for catching regressions
quickly; it does **not** replace the real acceptance checks on the target machine.

Two environment variables exist for debugging: `THREATMAMBA_FAKE_ENCODER=1` (deterministic
fake embeddings instead of SecureBERT) and `THREATMAMBA_SSM=simple` (pure-PyTorch SSM
instead of mamba-ssm — also the production fallback).

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `torch.cuda.is_available() == False` | Update the NVIDIA driver on Windows; `wsl --shutdown` and reopen; reinstall: `pip install --no-cache-dir torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128` |
| `no kernel image is available… sm_120` when calling Mamba | The installed `mamba-ssm` has no Blackwell kernel (prebuilt wheels stop at sm_100). Remove it (`pip uninstall mamba-ssm causal-conv1d`) and run `bash scripts/setup_mamba.sh`, or use `THREATMAMBA_SSM=simple` |
| `NameError: name 'bare_metal_version' is not defined` | The build cannot find `nvcc` → install CUDA Toolkit ≥ 12.8 and use `--no-build-isolation`. `scripts/setup_mamba.sh` handles this |
| `x86_64-linux-gnu-g++ (15.x) is greater than the maximum required version by CUDA` | CUDA rejects g++ 15 → install `g++-14`/`g++-13` and set `CUDAHOSTCXX`. `scripts/setup_mamba.sh` handles this |
| `exception specification is incompatible with that of previous function` | glibc ≥ 2.41 vs the CUDA headers → `sudo python3 scripts/patch_cuda_glibc.py` |
| `Precompiled wheel not found`, then a build failure | No wheel for your Python/torch combination (most often Python 3.14) → reinstall the environment with `scripts/setup_env.sh` to get Python 3.11 |
| `Command 'pip' not found` | The venv is not active: `source .venv/bin/activate` |
| The venv's Python is 3.14 | `bash scripts/reset_env.sh && bash scripts/setup_env.sh` — the script forces 3.11 |
| ollama `connection refused` | `ollama serve` or `sudo systemctl start ollama` |
| qwen3:8b keeps returning malformed JSON | It already retries 3 times; update Ollama. If it persists, add few-shot examples to the prompt in `src/ioc_hunter/llm_agent.py`, or swap in a similarly sized model |
| VT 401 / OTX 401-403 | Wrong key, or the account's email is not confirmed |
| Persistent VT 429 | Daily quota exhausted; the daemon logs it, skips, and continues the next day — no action needed |
| RapidDNS failing | Temporarily blocked by Cloudflare; retry later. Nothing else is affected |
| CSKG has only ~13 nodes per graph | Stage 3 was not trained, or the build used `--ttp none` → train Stage 3 and rebuild |
| Macro-F1 far too low | Check for leakage and label-map errors → run `--no-cl` to see the plain BCE baseline → consider reducing to the 10–12 best-represented groups |
| `main` clearly worse than `no_cl` | λ is too high and CL is swamping BCE. See [Three bugs](#three-bugs-worth-knowing-about); run `bash scripts/grid_cl.sh` |
| Training log: `cl` column stuck near **1.6094** | InfoNCE is at chance level (ln(K+1), K=4) — CL is learning nothing. Lower `cl_temperature` or raise `cl_pairs_K`; if it persists, the negatives are too easy |
| Training log: `pred 3/20` | Near-degenerate model. Lower `cl_lambda`, train longer, and check the CSKG actually has TTP nodes |
| Training stops too early (< 15 epochs) | Tune `early_stopping_patience` / `min_epochs` in `configs/default.yaml`; make sure every ablation runs through `scripts/train_all.sh` |
| Robustness table perfectly flat, A ≈ 0 | **Not** robustness — a degenerate model. See [Interpreting the output](#interpreting-the-output) |
| `Missing key(s) in state_dict: "ssm.layers.0.conv.weight"` … `Unexpected key(s): "ssm.layers.0.A_log"` | The SSM architecture does not match the checkpoint. `unset THREATMAMBA_SSM` and retry. That variable is for debugging and only applies when `model.ssm_fallback: auto` |
| UI says `outputs/model_main.pt` is missing | Run Stage 5 |
| UI still has a white background | Run `streamlit run` from the **repository root** so `.streamlit/config.toml` is read |
| UI pages 3/4 are empty | Run Stage 6 (`src.evaluate`, `src.explain`) |
| The pie-node graph does not render | ECharts is loaded from the jsdelivr CDN, so network access is required; check the browser console |
| `bash\r` errors | Files converted to CRLF: `sed -i 's/\r$//' scripts/*.sh scripts/checks/*.py` |

---

## Scope and deviations from the paper

This is an independent reproduction, not the authors' code. Known, deliberate deviations:

- **Language model.** The paper uses DeepSeek-R1-70B for IOCHunter-LLM; this uses
  **Qwen3-8B** via Ollama, which fits in 16 GB of VRAM.
- **Gaussian attention.** The per-label Gaussian prototype uses a **diagonal** Σ rather than
  a full covariance matrix.
- **Node contribution scores.** Eq. 21–23 are approximated by gradient × input on the node
  features with respect to the predicted class logit.
- **Candidate post-screening.** `iochunter.post_screen` adds an LLM screening pass over the
  candidates the APIs return. The paper has no such step (every hunting result enters the
  CSKG); it is enabled here because this corpus contains many benign reference domains. It
  can be switched off in `configs/default.yaml`.
- **Unspecified hyper-parameters.** The paper does not state λ (Eq. 18), τ (Eq. 20) or K.
  Defaults here are λ = 0.1, τ = 0.5, K = 4, selected via `scripts/grid_cl.sh`.
- **SSM fallback.** When `mamba-ssm` is unavailable the pure-PyTorch `SimpleSSM` is used.
  It is faithful in spirit but is not the paper's CUDA kernel; state this if you report
  numbers produced in that mode.
- **Table XI.** The Qi'anxin column of the paper's Table XI is omitted (that data source is
  not publicly available).

---

## Citation

If you use this repository, please cite the original paper:

```bibtex
@article{threatmamba2026,
  title   = {ThreatMAMBA},
  journal = {IEEE Transactions on Information Forensics and Security},
  year    = {2026},
  doi     = {10.1109/TIFS.2026.3685967}
}
```

and, optionally, this reproduction — see [CITATION.cff](CITATION.cff).

---

## Acknowledgements and licence

This work builds on:

- the **ThreatMAMBA** paper (IEEE TIFS 2026, DOI 10.1109/TIFS.2026.3685967);
- the dataset and the pie-node graph template from
  [`MuscleFish/ThreatMAMBA`](https://github.com/MuscleFish/ThreatMAMBA) (MIT) —
  `app/pie_node_graph.html` is that repository's `pie-node-ttp-state-graph.temp.html`,
  kept unmodified;
- [`ehsanaghaei/SecureBERT`](https://huggingface.co/ehsanaghaei/SecureBERT);
- [`state-spaces/mamba`](https://github.com/state-spaces/mamba) and
  [`Dao-AILab/causal-conv1d`](https://github.com/Dao-AILab/causal-conv1d);
- [MITRE ATT&CK v14](https://github.com/mitre-attack/attack-stix-data);
- [`ioc-finder`](https://github.com/ioc-fang/ioc-finder), [Ollama](https://ollama.com) /
  Qwen3, VirusTotal, AlienVault OTX and RapidDNS.

Code in this repository is released under the [MIT Licence](LICENSE). The datasets, models
and third-party services listed above retain their own licences and terms of use; check them
before redistributing data or results.

**Responsible use.** This is defensive security research tooling for analysing published
threat-intelligence reports. Threat attribution is probabilistic and error-prone — the
output of this system is an analytical aid, not evidence.
