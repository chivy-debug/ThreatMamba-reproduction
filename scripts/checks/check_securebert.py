#!/usr/bin/env python3
"""Stage 0 check: download SecureBERT (ehsanaghaei/SecureBERT) from Hugging Face and
encode three sample CTI sentences.

The first run downloads the model (~500 MB) into the cache (~/.cache/huggingface).

Exit codes: 0 = PASS, 1 = FAIL, 2 = SKIP (SMOKE_OFFLINE=1).
Run standalone:  python scripts/checks/check_securebert.py
"""
import os
import sys

MODEL_ID = "ehsanaghaei/SecureBERT"

SENTS = [
    "APT29 used spearphishing emails with malicious links to gain initial access.",
    "The malware communicates with its C2 server over HTTPS on port 443.",
    "Adversaries exfiltrated data using DNS tunneling to attacker-controlled domains.",
]


def fail(msg: str) -> int:
    print(f"[FAIL] securebert: {msg}")
    return 1


def main() -> int:
    if os.getenv("SMOKE_OFFLINE"):
        print("[SKIP] securebert: SMOKE_OFFLINE=1")
        return 2

    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
    except ImportError as e:
        return fail(f"missing dependency ({e}). Install torch (see check_cuda) and run: "
                    f"pip install -r requirements.txt")

    try:
        tok = AutoTokenizer.from_pretrained(MODEL_ID)
        model = AutoModel.from_pretrained(MODEL_ID)
    except Exception as e:
        return fail(
            f"failed to download the model from Hugging Face: {type(e).__name__}: {e}\n"
            "       Check your network/proxy. You can set HF_HOME to move the cache "
            "directory and retry."
        )

    model.eval()
    with torch.no_grad():
        enc = tok(SENTS, padding=True, truncation=True, max_length=64, return_tensors="pt")
        out = model(**enc).last_hidden_state   # (3, L, 768)
        emb = out.mean(dim=1)                  # mean-pool -> (3, 768)

    if tuple(emb.shape) != (3, 768):
        return fail(f"wrong embedding shape: {tuple(emb.shape)} (expected (3, 768))")

    print(f"[PASS] securebert: encoded 3 CTI sentences -> embedding {tuple(emb.shape)} OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
