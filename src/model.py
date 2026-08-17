"""Stage 5 classifier: relational GAT (Eq. 9-11) + MAMBA over the state sequence
(Eq. 12-14) + concatenation of the two branches (Eq. 15) + AvgPool||MaxPool (Eq. 16)
+ sigmoid MLP head (Eq. 17).

- Relational GAT: every relation type gets its own attention vector (Eq. 10); 4 heads,
  dim 48, 3 layers, residual + jump link. Implemented with plain PyTorch scatter ops,
  so torch_geometric is not required.
- MAMBA: SSMStack (official mamba-ssm, with the SimpleSSM fallback when unavailable).
- Ablation: no_mamba=True pools the state sequence directly, bypassing the SSM.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .cskg_builder import RELATIONS
from .ssm import SSMStack

N_RELS = len(RELATIONS) + 1  # +1: self-loop


class RelGATLayer(nn.Module):
    def __init__(self, d: int, heads: int, n_rels: int = N_RELS, dropout: float = 0.1):
        super().__init__()
        assert d % heads == 0
        self.d, self.h, self.dh = d, heads, d // heads
        self.W = nn.Linear(d, d, bias=False)
        self.a_src = nn.Parameter(torch.randn(n_rels, heads, self.dh) * 0.1)
        self.a_dst = nn.Parameter(torch.randn(n_rels, heads, self.dh) * 0.1)
        self.drop = nn.Dropout(dropout)
        self.bias = nn.Parameter(torch.zeros(d))

    def forward(self, h: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
        """h (N,d); edges (E,3) already include self-loops and reversed edges. Returns (N,d)."""
        N = h.shape[0]
        hw = self.W(h).view(N, self.h, self.dh)                      # (N,H,dh)
        u, v, r = edges[:, 0], edges[:, 1], edges[:, 2]
        e = (hw[u] * self.a_src[r]).sum(-1) + (hw[v] * self.a_dst[r]).sum(-1)  # (E,H)
        e = F.leaky_relu(e, 0.2)
        # softmax over the destination node v (scatter, numerically stable)
        e_max = torch.full((N, self.h), -1e30, device=h.device)
        e_max = e_max.index_reduce(0, v, e, "amax", include_self=True)
        ex = torch.exp(e - e_max[v])
        denom = torch.zeros(N, self.h, device=h.device).index_add(0, v, ex)
        alpha = self.drop(ex / denom[v].clamp(min=1e-12))            # (E,H)
        msg = hw[u] * alpha.unsqueeze(-1)                            # (E,H,dh)
        out = torch.zeros(N, self.h, self.dh, device=h.device).index_add(0, v, msg)
        return out.reshape(N, self.d) + self.bias


class RelGAT(nn.Module):
    def __init__(self, d: int, heads: int, n_layers: int, dropout: float):
        super().__init__()
        self.layers = nn.ModuleList(RelGATLayer(d, heads, N_RELS, dropout) for _ in range(n_layers))
        self.norms = nn.ModuleList(nn.LayerNorm(d) for _ in range(n_layers))
        self.jump = nn.Linear(d * n_layers, d)

    def forward(self, h: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
        outs = []
        for norm, layer in zip(self.norms, self.layers):
            h = h + F.elu(layer(norm(h), edges))                     # residual (Eq. 11)
            outs.append(h)
        return self.jump(torch.cat(outs, dim=-1))                    # jump link


def _prep_edges(edges: torch.Tensor, n_nodes: int, device) -> torch.Tensor:
    """Add reversed edges (same relation) and self-loops (their own relation id)."""
    if edges.numel():
        rev = torch.stack([edges[:, 1], edges[:, 0], edges[:, 2]], dim=1)
        e = torch.cat([edges, rev])
    else:
        e = torch.zeros(0, 3, dtype=torch.long)
    idx = torch.arange(n_nodes)
    self_loops = torch.stack([idx, idx, torch.full((n_nodes,), N_RELS - 1, dtype=torch.long)], dim=1)
    return torch.cat([e, self_loops]).to(device)


def _pool_masked(seq: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """seq (G,S,d), mask (G,S) -> AvgPool||MaxPool (G,2d)."""
    m = mask.unsqueeze(-1)
    avg = (seq * m).sum(1) / m.sum(1).clamp(min=1)
    mx = seq.masked_fill(~mask.bool().unsqueeze(-1), -1e30).max(1).values
    mx = torch.where(mask.sum(1, keepdim=True) > 0, mx, torch.zeros_like(mx))
    return torch.cat([avg, mx], dim=-1)


class ThreatMambaModel(nn.Module):
    def __init__(self, n_classes: int, d_in: int = 768, d: int = 48, heads: int = 4,
                 gat_layers: int = 3, ssm_layers: int = 3, dropout: float = 0.1,
                 ssm_mode: str = "auto", no_mamba: bool = False):
        super().__init__()
        self.no_mamba = no_mamba
        self.proj = nn.Linear(d_in, d)
        self.gat = RelGAT(d, heads, gat_layers, dropout)
        self.ssm = None if no_mamba else SSMStack(d, ssm_layers, ssm_mode, dropout)
        self.head = nn.Sequential(nn.Linear(4 * d, 128), nn.ReLU(), nn.Dropout(dropout),
                                  nn.Linear(128, n_classes))
        self.cl_proj = nn.Linear(4 * d, d)

    def forward(self, batch: dict, return_rep: bool = False):
        """batch: x (N,768), edges (E,3 original), graph_id (N,), states: list[list[int]]
        (globally offset). Returns (logits (G,C), z (G,d)), or (logits, z, rep (G,4d))
        when return_rep=True.

        `rep` is V_G from the paper: the whole-graph CSKG vector fed to the classification
        MLP (Eq. 17). `z` is the output of the contrastive projection head and is used ONLY
        for InfoNCE (Eq. 19-20).

        Eq. 24-26 (D_intra/D_inter/D_separ) and the t-SNE of Fig. 4 must be computed on
        `rep`, NOT on `z`: InfoNCE L2-normalises `z`, so it only retains ANGULAR information
        and loses magnitude. Euclidean distances measured on `z` diverge sharply from
        Table X of the paper."""
        x, edges, gid = batch["x"], batch["edges"], batch["graph_id"]
        G = int(batch["n_graphs"])
        device = x.device
        h = self.proj(x)                                              # (N,48)
        h = self.gat(h, _prep_edges(edges, x.shape[0], device))       # Eq. 9-11

        # GAT branch: pool every graph over all of its nodes
        d = h.shape[-1]
        avg = torch.zeros(G, d, device=device).index_add(0, gid, h)
        cnt = torch.zeros(G, device=device).index_add(0, gid, torch.ones_like(gid, dtype=h.dtype))
        avg = avg / cnt.clamp(min=1).unsqueeze(-1)
        mx = torch.full((G, d), -1e30, device=device).index_reduce(0, gid, h, "amax", include_self=True)
        gat_repr = torch.cat([avg, mx], dim=-1)                       # (G,2d) Eq. 16

        # MAMBA branch: the state sequence along the timeline (Eq. 12-14)
        S = max((len(s) for s in batch["states"]), default=1) or 1
        seq = torch.zeros(G, S, d, device=device)
        mask = torch.zeros(G, S, device=device)
        for gi, st in enumerate(batch["states"]):
            if st:
                seq[gi, :len(st)] = h[torch.tensor(st, device=device)]
                mask[gi, :len(st)] = 1
        if self.ssm is not None:
            seq = self.ssm(seq, mask)
        mamba_repr = _pool_masked(seq, mask)                          # (G,2d)

        rep = torch.cat([gat_repr, mamba_repr], dim=-1)               # (G,4d) Eq. 15-16 = V_G
        logits, z = self.head(rep), self.cl_proj(rep)                 # Eq. 17 + z for CL
        return (logits, z, rep) if return_rep else (logits, z)


def collate(graphs: list[dict], device, drop_rels: set[int] | None = None) -> dict:
    xs, edges, gids, states = [], [], [], []
    off = 0
    for gi, g in enumerate(graphs):
        n = len(g["node_types"])
        xs.append(g["x"].float())
        e = g["edges"]
        if drop_rels and e.numel():
            keep = torch.tensor([r not in drop_rels for r in e[:, 2].tolist()], dtype=torch.bool)
            e = e[keep]
        if e.numel():
            e = e.clone(); e[:, :2] += off
            edges.append(e)
        gids.append(torch.full((n,), gi, dtype=torch.long))
        states.append([s + off for s in g["states"]])
        off += n
    return {"x": torch.cat(xs).to(device) if xs else torch.zeros(0, 768, device=device),
            "edges": torch.cat(edges) if edges else torch.zeros(0, 3, dtype=torch.long),
            "graph_id": torch.cat(gids).to(device) if gids else torch.zeros(0, dtype=torch.long),
            "states": states, "n_graphs": len(graphs)}
