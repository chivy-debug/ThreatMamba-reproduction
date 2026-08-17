"""SSM stack shared by Stage 3 (TTP extraction) and Stage 5 (classifier).

- Default: the official `mamba-ssm` package, which requires a CUDA GPU.
- Fallback (only when mamba-ssm is unavailable, or for CPU tests): SimpleSSMBlock,
  a pure-PyTorch block (causal conv1d + selective gating + linear scan). Slower, but
  computationally equivalent in spirit; state this explicitly in any write-up.
  Selected via config model.ssm_fallback: auto|mamba|simple, or env THREATMAMBA_SSM=simple.
"""
import os

import torch
import torch.nn as nn

_WARNED = False


def _resolve(mode: str) -> str:
    global _WARNED
    env = os.getenv("THREATMAMBA_SSM")
    # IMPORTANT: the env var ONLY takes effect when mode == "auto".
    # When loading a checkpoint the architecture MUST match the one used at training time,
    # so evaluate.load_model passes the mode recorded in the checkpoint ("mamba"/"simple")
    # explicitly. Letting the env var override it would build the wrong architecture and
    # make load_state_dict raise "Missing/Unexpected key(s)".
    if mode not in ("mamba", "simple") and env in ("simple", "mamba"):
        mode = env
    if mode == "mamba":
        return "mamba"
    if mode == "simple":
        return "simple"
    # auto
    try:
        import mamba_ssm  # noqa: F401
        if torch.cuda.is_available():
            return "mamba"
        reason = "no CUDA device"
    except ImportError:
        reason = "mamba-ssm not installed"
    if not _WARNED:
        print(f"[ssm] FALLBACK: pure-PyTorch SimpleSSM ({reason})")
        _WARNED = True
    return "simple"


class SimpleSSMBlock(nn.Module):
    """Minimal SSM block: depthwise causal conv -> SiLU -> selective scan
    h_t = a_t*h_{t-1} + (1-a_t)*x_t  with a_t = sigmoid(W_a x_t)  -> output gate."""

    def __init__(self, d_model: int, d_conv: int = 4):
        super().__init__()
        self.conv = nn.Conv1d(d_model, d_model, d_conv, padding=d_conv - 1, groups=d_model)
        self.w_a = nn.Linear(d_model, d_model)
        self.w_g = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B,S,D)
        B, S, D = x.shape
        c = self.conv(x.transpose(1, 2))[:, :, :S].transpose(1, 2)
        c = self.act(c)
        a = torch.sigmoid(self.w_a(c))
        h = torch.zeros(B, D, device=x.device, dtype=x.dtype)
        hs = []
        for t in range(S):
            h = a[:, t] * h + (1 - a[:, t]) * c[:, t]
            hs.append(h)
        hseq = torch.stack(hs, dim=1)
        return self.out(hseq * self.act(self.w_g(x)))


def make_ssm_layer(d_model: int, mode: str) -> nn.Module:
    if mode == "mamba":
        from mamba_ssm import Mamba
        return Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2)
    return SimpleSSMBlock(d_model)


class SSMStack(nn.Module):
    """n_layers SSM blocks, each pre-norm + residual, plus a jump link from the
    input (Eq. 13-14)."""

    def __init__(self, d_model: int, n_layers: int = 3, mode: str = "auto", dropout: float = 0.1):
        super().__init__()
        self.mode = _resolve(mode)
        self.layers = nn.ModuleList(make_ssm_layer(d_model, self.mode) for _ in range(n_layers))
        self.norms = nn.ModuleList(nn.LayerNorm(d_model) for _ in range(n_layers))
        self.drop = nn.Dropout(dropout)
        self.jump = nn.Linear(2 * d_model, d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """x (B,S,D), mask (B,S) with 1 = real token. Returns (B,S,D)."""
        x0 = x
        for norm, layer in zip(self.norms, self.layers):
            x = x + self.drop(layer(norm(x)))
            if mask is not None:
                x = x * mask.unsqueeze(-1)
        x = self.jump(torch.cat([x, x0], dim=-1))
        if mask is not None:
            x = x * mask.unsqueeze(-1)
        return x
