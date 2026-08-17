"""Stage 5 losses: BCE (Eq. 18) + InfoNCE-style contrastive loss (Eq. 19-20).
L = L_BCE + lambda * L_CL
"""
import torch
import torch.nn.functional as F


def bce_loss(logits: torch.Tensor, y_onehot: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(logits, y_onehot)


def info_nce(z: torch.Tensor, z_pos: torch.Tensor, z_neg: torch.Tensor,
             tau: float = 0.5) -> torch.Tensor:
    """z, z_pos: (G,d); z_neg: (G,K,d). Cosine similarity scaled by tau (Eq. 20)."""
    z = F.normalize(z, dim=-1)
    z_pos = F.normalize(z_pos, dim=-1)
    z_neg = F.normalize(z_neg, dim=-1)
    s_pos = (z * z_pos).sum(-1) / tau                     # (G,)
    s_neg = torch.einsum("gd,gkd->gk", z, z_neg) / tau    # (G,K)
    logits = torch.cat([s_pos.unsqueeze(1), s_neg], dim=1)
    target = torch.zeros(z.shape[0], dtype=torch.long, device=z.device)
    return F.cross_entropy(logits, target)
