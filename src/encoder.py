"""Sentence encoder: SecureBERT (RoBERTa-base, mean-pooled tokens) + on-disk cache.

- Used frozen, NOT fine-tuned.
- FakeEncoder: deterministic sha1-derived embeddings, for CPU tests ONLY
  (enabled with THREATMAMBA_FAKE_ENCODER=1). Never cached to disk.
"""
import hashlib
import os

import numpy as np

from .utils import PROCESSED, torch_load

EMB_DIM = 768
CACHE = PROCESSED / "emb_cache"
MODEL_ID = "ehsanaghaei/SecureBERT"


class FakeEncoder:
    dim = EMB_DIM
    is_fake = True

    def encode(self, sents: list[str]):
        import torch
        if not sents:
            return torch.zeros(0, EMB_DIM)
        arr = np.stack([self._one(s) for s in sents])
        return torch.from_numpy(arr)

    @staticmethod
    def _one(s: str):
        h = hashlib.sha1(s.encode("utf-8", "ignore")).digest()
        rng = np.random.default_rng(int.from_bytes(h[:8], "little"))
        return rng.standard_normal(EMB_DIM).astype(np.float32)


class SecureBERTEncoder:
    is_fake = False

    def __init__(self, device=None, batch_size: int = 32, max_length: int = 128):
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(MODEL_ID)
        self.model = AutoModel.from_pretrained(MODEL_ID).eval()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.batch_size = batch_size
        self.max_length = max_length
        self.dim = EMB_DIM

    def encode(self, sents: list[str]):
        """(N sentences) -> float32 CPU tensor (N, 768), mean-pooled over the attention mask."""
        import torch
        if not sents:
            return torch.zeros(0, EMB_DIM)
        outs = []
        with torch.no_grad():
            for i in range(0, len(sents), self.batch_size):
                batch = [s[:2000] for s in sents[i:i + self.batch_size]]
                enc = self.tok(batch, padding=True, truncation=True,
                               max_length=self.max_length, return_tensors="pt").to(self.device)
                hid = self.model(**enc).last_hidden_state
                mask = enc["attention_mask"].unsqueeze(-1)
                emb = (hid * mask).sum(1) / mask.sum(1).clamp(min=1)
                outs.append(emb.float().cpu())
        return torch.cat(outs)


def get_encoder(device=None):
    if os.getenv("THREATMAMBA_FAKE_ENCODER"):
        return FakeEncoder()
    return SecureBERTEncoder(device=device)


def encode_doc_cached(doc_id: str, sents: list[str], encoder):
    """Encode the sentences of one document, cached as .pt keyed by (doc_id, content hash)."""
    import torch
    if getattr(encoder, "is_fake", False):
        return encoder.encode(sents)
    CACHE.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha1("\n".join(sents).encode("utf-8", "ignore")).hexdigest()[:16]
    fp = CACHE / f"{doc_id}_{key}.pt"
    if fp.exists():
        return torch_load(fp, weights_only=True)
    emb = encoder.encode(sents)
    torch.save(emb, fp)
    return emb
