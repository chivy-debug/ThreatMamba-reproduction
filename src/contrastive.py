"""Stage 5 contrastive sampling (Eq. 7-8).

- POSITIVE view (Eq. 7): the same graph with Gaussian feature noise and Bernoulli
  edge dropping (the temporal_next backbone is preserved so the timeline stays intact).
- NEGATIVE view (Eq. 8): splice the timelines of two events with DIFFERENT labels using
  an offset phi in (-100,100)\\{0}, then rewire the ATT&CK relations (kill-chain
  tactic_seq) over the merged graph.
"""
import random

import torch

from .cskg_builder import NODE_TYPES, NT, REL


def pos_view(g: dict, noise_std: float = 0.05, edge_drop: float = 0.1,
             rng: random.Random | None = None) -> dict:
    rng = rng or random
    x = g["x"].float()
    x = x + torch.randn_like(x) * noise_std
    keep_rows = []
    for row in g["edges"].tolist():
        if row[2] == REL["temporal_next"] or rng.random() >= edge_drop:
            keep_rows.append(row)
    edges = torch.tensor(keep_rows, dtype=torch.long) if keep_rows else torch.zeros(0, 3, dtype=torch.long)
    return {**g, "x": x, "edges": edges}


def neg_splice(g1: dict, g2: dict, tac_order: list[str], phi: int | None = None,
               rng: random.Random | None = None) -> dict:
    """Splice g1 (anchor) with g2 (different label): shift g2's timeline by phi
    sentences, then merge the two graphs."""
    rng = rng or random
    if phi is None:
        phi = 0
        while phi == 0:
            phi = rng.randint(-99, 99)
    off = len(g1["node_types"])
    node_types = list(g1["node_types"]) + list(g2["node_types"])
    node_values = list(g1["node_values"]) + [v + "#2" for v in g2["node_values"]]
    node_sent = list(g1["node_sent"]) + [s + phi for s in g2["node_sent"]]
    x = torch.cat([g1["x"].float(), g2["x"].float()])

    edges = []
    for u, v, r in g1["edges"].tolist():
        if r not in (REL["temporal_next"], REL["tactic_seq"]):
            edges.append((u, v, r))
    for u, v, r in g2["edges"].tolist():
        if r not in (REL["temporal_next"], REL["tactic_seq"]):
            edges.append((u + off, v + off, r))

    # new temporal_next backbone: merge states by their shifted node_sent
    merged_states = sorted([s for s in g1["states"]] + [s + off for s in g2["states"]],
                           key=lambda n: node_sent[n])
    for a, b in zip(merged_states, merged_states[1:]):
        edges.append((a, b, REL["temporal_next"]))

    # rewire tactic_seq following the ATT&CK kill-chain order over the merged tactics
    tac_nodes: dict[str, int] = {}
    for i, (t, v) in enumerate(zip(node_types, node_values)):
        if NODE_TYPES[t] == "Tactic":
            tac_nodes.setdefault(v.replace("#2", ""), i)
    chain = [s for s in tac_order if s in tac_nodes]
    for a, b in zip(chain, chain[1:]):
        edges.append((tac_nodes[a], tac_nodes[b], REL["tactic_seq"]))

    return {"doc_id": g1["doc_id"] + "|neg|" + g2["doc_id"], "group": g1["group"],
            "node_types": node_types, "node_values": node_values, "node_sent": node_sent,
            "x": x, "edges": torch.tensor(edges, dtype=torch.long) if edges else torch.zeros(0, 3, dtype=torch.long),
            "states": merged_states, "n_sents": max(g1.get("n_sents", 0), g2.get("n_sents", 0) + phi)}
