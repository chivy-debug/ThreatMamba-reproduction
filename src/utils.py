"""Shared helpers: config, seed, device, text cleaning, sentence split, jsonl IO."""
import json
import random
import re
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW = DATA / "raw"
PROCESSED = DATA / "processed"
CSKG_DIR = DATA / "cskg"
ENRICHED = DATA / "enriched"
ATTCK_DIR = DATA / "attck"
REFERENCE = DATA / "reference"
OUTPUTS = ROOT / "outputs"


def load_config(path=None):
    import yaml
    p = Path(path) if path else ROOT / "configs" / "default.yaml"
    with open(p, encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int = 42):
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def get_device(pref: str | None = None):
    import torch
    if pref:
        return torch.device(pref)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


_WS = re.compile(r"\s+")
_CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def clean_text(s) -> str:
    """Clean text coming from the source CSVs (which contain real encoding damage):
    - the CSVs are read with encoding_errors='replace', so broken bytes become U+FFFD
    - a run of U+FFFD is replaced by an apostrophe (by far the most common case:
      don't, it's, ...)
    - NFKC normalise, strip control characters, collapse whitespace."""
    if not isinstance(s, str):
        return ""
    s = unicodedata.normalize("NFKC", s)
    s = re.sub("�+", "'", s)
    s = _CTRL.sub(" ", s)
    return _WS.sub(" ", s).strip()


_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")


def sent_split(text: str, min_len: int = 15, max_sents: int = 256) -> list[str]:
    """Simple deterministic sentence splitter, no nltk required.
    Very short fragments are merged into the preceding sentence."""
    text = (text or "").strip()
    if not text:
        return []
    parts = [p.strip() for p in _SENT.split(text) if p.strip()]
    out: list[str] = []
    for p in parts:
        if len(p) >= min_len or not out:
            out.append(p)
        else:
            out[-1] = out[-1] + " " + p
    return out[:max_sents]


def read_jsonl(path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_docs(split: str | None = None) -> list[dict]:
    """Read data/processed/docs.jsonl, optionally filtered by split ('train'/'test')."""
    rows = read_jsonl(PROCESSED / "docs.jsonl")
    if split:
        rows = [r for r in rows if r.get("split") == split]
    return rows


def save_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def torch_load(path, map_location="cpu", weights_only: bool = False):
    """Backwards-compatible torch.load: torch < 1.13 has no weights_only argument."""
    import torch
    try:
        return torch.load(path, map_location=map_location, weights_only=weights_only)
    except TypeError:
        return torch.load(path, map_location=map_location)
