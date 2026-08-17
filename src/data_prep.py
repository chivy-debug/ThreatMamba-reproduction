"""Stage 1: data preparation.

Subcommands:
  python -m src.data_prep prepare      # D2: clean, keep groups with >=30 docs, 8:2 split
                                       #     -> docs.jsonl + manifest.csv
  python -m src.data_prep prepare-d1   # D1 (CTI2TTPs): text + multi-hot labels
                                       #     -> d1.jsonl + label space
  python -m src.data_prep attck        # parse enterprise-attack-14.1.json -> attck_v14.json
  python -m src.data_prep demo-subset  # pick 30-50 demo documents (IOC-rich, spread over groups)
  python -m src.data_prep stats        # Stage 1 acceptance check: print dataset statistics

Actual schema (verified against the MuscleFish/ThreatMAMBA repository):
  CTI2Attacker.csv  : group, group_name, cti_content
                      (10,510 rows, 1,149 groups, genuine UTF-8 damage)
  CTI2TTPs.csv      : Text + 14 TA#### columns + 541 T####(.###) columns, 0/1 labels
                      (12,406 samples)
  CTI2Attacker_TTP.csv: TTPs/Context/Group/original_index. original_index does NOT match the
                      row order of CTI2Attacker.csv (verified), so it is used for reference
                      only and never joined.
"""
import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path

from .utils import (ATTCK_DIR, PROCESSED, RAW, clean_text, load_config,
                    read_jsonl, save_json, sent_split, write_jsonl)

D2_CSV = "CTI2Attacker.csv"
D1_CSV = "CTI2TTPs.csv"
ATTCK_JSON = "enterprise-attack-14.1.json"


def _raw_dir(args) -> Path:
    return Path(args.raw_dir) if args.raw_dir else RAW / "MuscleFish"


# ----------------------------------------------------------------- D2
def cmd_prepare(args):
    import pandas as pd
    cfg = load_config(args.config)
    min_docs = int(cfg["data"]["min_docs_per_group"])
    ratio = float(cfg["data"]["train_test_ratio"])
    seed = int(cfg["data"].get("seed", 42))
    min_chars = int(cfg["data"].get("min_doc_chars", 200))

    src = _raw_dir(args) / D2_CSV
    df = pd.read_csv(src, usecols=["group", "group_name", "cti_content"],
                     encoding="utf-8", encoding_errors="replace")
    df["text"] = df["cti_content"].map(clean_text)
    df["doc_id"] = [f"d{i:05d}" for i in range(len(df))]
    n0 = len(df)
    df = df[df["text"].str.len() >= min_chars]
    print(f"Read {n0} rows -> {len(df)} after dropping short documents (<{min_chars} chars)")

    vc = df["group_name"].value_counts()
    keep = sorted(vc[vc >= min_docs].index.tolist())
    df = df[df["group_name"].isin(keep)]
    print(f"Groups with >= {min_docs} documents: kept {len(keep)} groups, {len(df)} documents")

    rng = random.Random(seed)
    rows, manifest = [], []
    for g in keep:
        ids = sorted(df[df["group_name"] == g]["doc_id"].tolist())
        rng.shuffle(ids)
        n_train = max(1, int(round(len(ids) * ratio)))
        if n_train == len(ids):
            n_train -= 1  # guarantee every group keeps at least one test document
        train_ids = set(ids[:n_train])
        for did in ids:
            manifest.append({"doc_id": did, "group": g,
                             "split": "train" if did in train_ids else "test"})
    split_of = {m["doc_id"]: m["split"] for m in manifest}
    for _, r in df.iterrows():
        rows.append({"doc_id": r["doc_id"], "group": r["group_name"],
                     "split": split_of[r["doc_id"]], "text": r["text"]})

    PROCESSED.mkdir(parents=True, exist_ok=True)
    write_jsonl(PROCESSED / "docs.jsonl", rows)
    pd.DataFrame(manifest).to_csv(PROCESSED / "manifest.csv", index=False)
    save_json(PROCESSED / "groups.json", keep)
    n_tr = sum(1 for m in manifest if m["split"] == "train")
    print(f"OK: docs.jsonl + manifest.csv | train={n_tr} test={len(manifest) - n_tr} | groups={len(keep)}")


# ----------------------------------------------------------------- D1
def cmd_prepare_d1(args):
    import pandas as pd
    df = pd.read_csv(_raw_dir(args) / D1_CSV, encoding="utf-8", encoding_errors="replace")
    label_cols = [c for c in df.columns if re.fullmatch(r"TA?\d{4}(\.\d{3})?", c)]
    tactics = [c for c in label_cols if c.startswith("TA")]
    techniques = [c for c in label_cols if not c.startswith("TA")]
    rows = []
    for _, r in df.iterrows():
        text = clean_text(r["Text"])
        if len(text) < 30:
            continue
        labels = [c for c in label_cols if int(r[c]) == 1]
        rows.append({"text": text, "labels": labels})
    write_jsonl(PROCESSED / "d1.jsonl", rows)
    save_json(PROCESSED / "d1_label_space.json",
              {"tactics": tactics, "techniques": techniques})
    pos = Counter(l for r in rows for l in r["labels"])
    print(f"D1: {len(rows)} samples | {len(tactics)} tactics | {len(techniques)} technique columns (incl. sub-techniques)")
    print(f"    techniques with >=5 positive samples: {sum(1 for c in techniques if pos[c] >= 5)}")


# ----------------------------------------------------------------- ATT&CK v14
def cmd_attck(args):
    """Parse the STIX bundle with plain json (lighter than mitreattack-python, same result)."""
    src = Path(args.attck_json) if args.attck_json else ATTCK_DIR / ATTCK_JSON
    with open(src, encoding="utf-8") as f:
        bundle = json.load(f)
    objs = bundle["objects"]
    by_id = {o["id"]: o for o in objs}

    def ext_id(o):
        for ref in o.get("external_references", []):
            if ref.get("source_name") == "mitre-attack":
                return ref.get("external_id")
        return None

    def alive(o):
        return not o.get("revoked") and not o.get("x_mitre_deprecated")

    # tactic order taken from the matrix object (kill-chain order)
    matrix = next(o for o in objs if o["type"] == "x-mitre-matrix")
    tactic_order = []
    for tid in matrix["tactic_refs"]:
        t = by_id[tid]
        tactic_order.append({"id": ext_id(t), "shortname": t["x_mitre_shortname"],
                             "name": t["name"]})

    techniques = {}
    for o in objs:
        if o["type"] == "attack-pattern" and alive(o):
            tid = ext_id(o)
            if not tid:
                continue
            phases = [p["phase_name"] for p in o.get("kill_chain_phases", [])
                      if p.get("kill_chain_name") == "mitre-attack"]
            techniques[tid] = {"name": o["name"], "tactics": phases,
                               "stix_id": o["id"],
                               "is_sub": o.get("x_mitre_is_subtechnique", False)}
    stix2tid = {v["stix_id"]: k for k, v in techniques.items()}

    groups = {}
    gid_by_stix = {}
    for o in objs:
        if o["type"] == "intrusion-set" and alive(o):
            groups[o["name"]] = {"attck_id": ext_id(o),
                                 "aliases": o.get("aliases", []), "techniques": []}
            gid_by_stix[o["id"]] = o["name"]
    for o in objs:
        if (o["type"] == "relationship" and o.get("relationship_type") == "uses"
                and o["source_ref"] in gid_by_stix and o["target_ref"] in stix2tid
                and alive(o)):
            groups[gid_by_stix[o["source_ref"]]]["techniques"].append(stix2tid[o["target_ref"]])
    for g in groups.values():
        g["techniques"] = sorted(set(g["techniques"]))

    out = {"version": "v14", "tactic_order": tactic_order,
           "techniques": techniques, "groups": groups}
    save_json(ATTCK_DIR / "attck_v14.json", out)
    print(f"ATT&CK v14: {len(techniques)} techniques (incl. sub-techniques) | "
          f"{len(tactic_order)} tactics | {len(groups)} groups -> data/attck/attck_v14.json")


# ----------------------------------------------------------------- demo subset
def cmd_demo_subset(args):
    from .ioc_extract import extract_iocs
    docs = read_jsonl(PROCESSED / "docs.jsonl")
    per_group: dict[str, list] = {}
    for d in docs:
        n_ioc = len(extract_iocs(d["text"]))
        per_group.setdefault(d["group"], []).append((n_ioc, d["split"], d["doc_id"]))
    chosen = []
    for g in sorted(per_group):
        # prefer IOC-rich documents; at most 2 per group
        ranked = sorted(per_group[g], key=lambda t: (-t[0], t[2]))
        chosen += [(g, did, n) for n, _s, did in ranked[:2] if n >= 3]
    chosen = chosen[: args.max_docs]
    lines = [did for _g, did, _n in chosen]
    (PROCESSED.parent / "demo_subset.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"demo_subset.txt: {len(lines)} documents from {len({g for g, _, _ in chosen})} groups")
    for g, did, n in chosen[:10]:
        print(f"   {did}  {g:<15} {n} IOC")


# ----------------------------------------------------------------- stats (acceptance)
def cmd_stats(args):
    import pandas as pd
    print("=" * 60)
    print("STAGE 1 ACCEPTANCE - DATASET STATISTICS")
    print("=" * 60)
    docs = read_jsonl(PROCESSED / "docs.jsonl")
    df = pd.DataFrame([{k: d[k] for k in ("doc_id", "group", "split")} | {"len": len(d["text"])}
                       for d in docs])
    print(f"[D2] {len(df)} documents | {df.group.nunique()} groups | "
          f"train={sum(df.split == 'train')} test={sum(df.split == 'test')}")
    tbl = df.groupby("group").agg(n=("doc_id", "count"), len_tb=("len", "mean")).sort_values("n", ascending=False)
    print(tbl.to_string(float_format=lambda x: f"{x:,.0f}"))
    q = df["len"].quantile([.1, .5, .9]).astype(int)
    print(f"Document length in characters, p10/p50/p90: {q.iloc[0]:,}/{q.iloc[1]:,}/{q.iloc[2]:,}")

    d1p = PROCESSED / "d1.jsonl"
    if d1p.exists():
        d1 = read_jsonl(d1p)
        space = json.load(open(PROCESSED / "d1_label_space.json", encoding="utf-8"))
        print(f"\n[D1] {len(d1)} samples | tactics={len(space['tactics'])} | "
              f"technique cols={len(space['techniques'])}")

    ap = ATTCK_DIR / "attck_v14.json"
    if ap.exists():
        a = json.load(open(ap, encoding="utf-8"))
        n_tech, n_tac = len(a["techniques"]), len(a["tactic_order"])
        ok = "PASS" if (n_tech >= 600 and n_tac == 14) else "FAIL"
        print(f"\n[ATT&CK v14] techniques={n_tech} (need >=600) | tactics={n_tac} (need 14) "
              f"| groups={len(a['groups'])}  -> {ok}")

    dsp = PROCESSED.parent / "demo_subset.txt"
    if dsp.exists():
        n = len([l for l in dsp.read_text().splitlines() if l.strip()])
        print(f"\n[Demo subset] {n} documents (target 30-50)")
    print("=" * 60)


def main():
    ap = argparse.ArgumentParser(description="Stage 1 - data preparation")
    ap.add_argument("cmd", choices=["prepare", "prepare-d1", "attck", "demo-subset", "stats", "all"])
    ap.add_argument("--raw-dir", default=None,
                    help="directory holding the MuscleFish CSVs (default data/raw/MuscleFish)")
    ap.add_argument("--attck-json", default=None)
    ap.add_argument("--config", default=None)
    ap.add_argument("--max-docs", type=int, default=40)
    args = ap.parse_args()
    steps = {"prepare": cmd_prepare, "prepare-d1": cmd_prepare_d1, "attck": cmd_attck,
             "demo-subset": cmd_demo_subset, "stats": cmd_stats}
    if args.cmd == "all":
        for name in ["prepare", "prepare-d1", "attck", "demo-subset", "stats"]:
            print(f"\n### {name}")
            steps[name](args)
    else:
        steps[args.cmd](args)


if __name__ == "__main__":
    main()
