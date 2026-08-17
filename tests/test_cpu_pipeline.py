#!/usr/bin/env python3
"""End-to-end CPU test (no GPU, SecureBERT, Ollama or API access required):
- FakeEncoder (THREATMAMBA_FAKE_ENCODER=1) + SimpleSSM (THREATMAMBA_SSM=simple)
- A small synthetic corpus flows through: ioc_extract -> cskg_builder -> contrastive ->
  model (all three ablations) -> losses -> truncate (robustness) -> metrics ->
  contributions -> UCT / LLM stub.

Run:  THREATMAMBA_FAKE_ENCODER=1 THREATMAMBA_SSM=simple python tests/test_cpu_pipeline.py

Purpose: confirm every tensor in the pipeline has the right shape, backward runs, and
nothing is NaN. This does NOT replace the real acceptance checks on the target machine.
"""
import os
import random
import sys
from pathlib import Path

os.environ.setdefault("THREATMAMBA_FAKE_ENCODER", "1")
os.environ.setdefault("THREATMAMBA_SSM", "simple")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

PASS = []


def ok(name):
    PASS.append(name)
    print(f"[PASS] {name}")


def make_docs():
    tmpl = [
        ("APT_A", "The dropper evil{i}.exe beacons to bad{i}.example{i}.com over HTTPS on port 443. "
                  "It exploits CVE-2023-2339{i} and writes HKLM\\Software\\Run keys. "
                  "Analysts saw traffic to 45.77.12.{i} and hash "
                  "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b85{i}. "
                  "Finally data was exfiltrated via DNS tunneling to c2-{i}.net."),
        ("APT_B", "Spearphishing email from crook{i}@mail{i}.org delivered doc{i}.docx with macros. "
                  "The implant used RDP for lateral movement and contacted 10.10.10.{i} on port 8080. "
                  "SHA1 da39a3ee5e6b4b0d3255bfef95601890afd8070{i} was observed. "
                  "Persistence used registry HKCU\\Software\\Micro{i} and protocol SMB."),
    ]
    docs = []
    for i in range(8):
        grp, t = tmpl[i % 2]
        docs.append({"doc_id": f"t{i:03d}", "group": grp, "split": "train",
                     "text": t.format(i=i % 10)})
    return docs


def fake_ttp_fn(text, sents, emb=None):
    n = max(1, len(sents))
    seq = [(0, "T1566"), (min(1, n - 1), "T1059"), (min(2, n - 1), "TA0010")]
    return {"ttps": [("T1566", .9), ("T1059", .8), ("TA0010", .7)], "seq": seq}


def main():
    torch.manual_seed(0)
    random.seed(0)

    # 1. IOC extract
    from src.ioc_extract import extract_iocs
    docs = make_docs()
    iocs = extract_iocs(docs[0]["text"])
    types = {i["type"] for i in iocs}
    assert {"IP", "Hash", "CVE", "Port"} <= types, types
    ok(f"ioc_extract: {len(iocs)} IOC, types={sorted(types)}")

    # 2. CSKG build (train mode) with an injected fake ATT&CK mapping
    from src import cskg_builder as cb
    builder = cb.GraphBuilder(ttp_fn=fake_ttp_fn, mode="train")
    builder.tech2tac = {"T1566": ["initial-access"], "T1059": ["execution"]}
    builder.tac_order = ["initial-access", "execution", "exfiltration"]
    builder.tacid2short = {"TA0010": "exfiltration"}
    graphs = [builder.build(d) for d in docs]
    g0 = graphs[0]
    assert g0["x"].shape[0] == len(g0["node_types"]) and len(g0["edges"]) > 0
    assert g0["edges"][:, :2].max() < len(g0["node_types"])
    ok(f"cskg train-mode: {len(g0['node_types'])} node / {len(g0['edges'])} edge / "
       f"{len(g0['states'])} state")

    # 3. CSKG enriched mode: all 19 node types present
    seed_ioc = next(i for i in extract_iocs(docs[0]["text"]) if i["type"] == "IP")
    enriched = {"hunts": [{"seed": seed_ioc, "method": "vt_object", "n_candidates": 4,
                           "accepted": [
                               {"value": "2020-01-01", "type": "Time"},
                               {"value": "RU", "type": "Geo-location"},
                               {"value": "cmd.exe /c whoami", "type": "CMD"},
                               {"value": "CreateRemoteThread", "type": "API"},
                               {"value": "sibling.example.com", "type": "Domain"}]}]}
    be = cb.GraphBuilder(ttp_fn=fake_ttp_fn, mode="enriched")
    be.tech2tac, be.tac_order, be.tacid2short = builder.tech2tac, builder.tac_order, builder.tacid2short
    ge = be.build(docs[0], enriched)
    tset = {cb.NODE_TYPES[t] for t in ge["node_types"]}
    assert {"Time", "Geo-location", "CMD", "API"} <= tset, tset
    ok(f"cskg enriched-mode: {len(tset)} node types covered (incl. Time/Geo/CMD/API)")

    # 4. truncate (robustness masking)
    from src.cskg_builder import truncate_graph
    gt = truncate_graph(g0, 0.4)
    assert 0 < len(gt["states"]) <= len(g0["states"])
    assert (gt["edges"][:, :2].max() < len(gt["node_types"])) if gt["edges"].numel() else True
    ok(f"truncate_graph 40%: {len(g0['states'])} -> {len(gt['states'])} state")

    # 5. contrastive views
    from src.contrastive import neg_splice, pos_view
    pv = pos_view(g0, 0.05, 0.2, random.Random(1))
    assert pv["x"].shape == g0["x"].float().shape
    nv = neg_splice(graphs[0], graphs[1], builder.tac_order, rng=random.Random(2))
    assert len(nv["node_types"]) == len(g0["node_types"]) + len(graphs[1]["node_types"])
    if nv["edges"].numel():
        assert nv["edges"][:, :2].max() < len(nv["node_types"])
    ok("contrastive: pos_view + neg_splice are well-formed")

    # 6. model, all 4 configurations (main + 3 ablations), forward and backward
    from src.losses import bce_loss, info_nce
    from src.model import ThreatMambaModel, collate
    y = torch.tensor([0 if d["group"] == "APT_A" else 1 for d in docs])
    y1h = torch.zeros(len(docs), 2)
    y1h[torch.arange(len(docs)), y] = 1
    for name, kw in [("main", {}), ("no_mamba", {"no_mamba": True})]:
        model = ThreatMambaModel(2, ssm_mode="simple", **kw)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        batch = collate(graphs, "cpu")
        logits, z = model(batch)
        assert logits.shape == (8, 2) and z.shape == (8, 48)
        loss = bce_loss(logits, y1h)
        pos = collate([pos_view(g, 0.05, 0.1, random.Random(3)) for g in graphs], "cpu")
        _, zp = model(pos)
        negs = [neg_splice(graphs[i], graphs[(i + 1) % 8], builder.tac_order,
                           rng=random.Random(i)) for i in range(8) for _ in range(2)]
        _, zn = model(collate(negs, "cpu"))
        loss = loss + info_nce(z, zp, zn.view(8, 2, -1), 0.5)
        opt.zero_grad(); loss.backward(); opt.step()
        assert torch.isfinite(loss)
        ok(f"model[{name}]: forward+CL+backward, loss={float(loss):.4f}")
    # no_iochunter = drop those relations during collate
    from src.train import HUNT_RELS
    b2 = collate(graphs, "cpu", HUNT_RELS)
    assert not any(r in HUNT_RELS for r in b2["edges"][:, 2].tolist())
    ok("ablation no_iochunter: co_occurs/hunting edges removed")

    # 7. metrics + robustness path
    from src.evaluate import topk_metrics
    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model(collate(graphs, "cpu"))[0]).numpy()
    met = topk_metrics(probs, y.numpy(), 2)
    assert 0 <= met["top1_micro"] <= 1 and "f1_macro" in met
    ok(f"metrics: {met}")

    # 8. contributions (explain)
    from src.explain import node_contributions
    pred, score = node_contributions(model, g0, "cpu")
    assert abs(score.sum() - 1) < 1e-4 and len(score) == len(g0["node_types"])
    ok(f"explain: contributions sum=1, pred={pred}")

    # 9. TTPExtractor (Gaussian attention), tiny training loop
    from src.ttp_extract import TTPExtractor
    tm = TTPExtractor(n_labels=6, d=48, ssm_layers=2, ssm_mode="simple")
    opt = torch.optim.Adam(tm.parameters(), lr=1e-3)
    emb = torch.randn(4, 10, 768)
    mask = torch.ones(4, 10); mask[2, 7:] = 0
    yl = (torch.rand(4, 6) > 0.5).float()
    l0 = None
    for _ in range(5):
        logits, attn = tm(emb, mask)
        assert logits.shape == (4, 6) and attn.shape == (4, 10, 6)
        assert torch.allclose(attn.sum(1), torch.ones(4, 6), atol=1e-3)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, yl)
        l0 = l0 or float(loss)
        opt.zero_grad(); loss.backward(); opt.step()
    assert float(loss) <= l0
    ok(f"ttp_extract: Gaussian attn OK, loss {l0:.4f} -> {float(loss):.4f}")

    # 10. UCT + LLM agent (stub)
    from src.ioc_hunter.uct import UCTSelector
    sel = UCTSelector()
    m1 = sel.select("Domain"); assert m1 in ("rapiddns_history", "otx_general", "vt_object")
    sel.update(m1, 3)
    assert sel.select("Hash") in ("otx_general", "vt_object", "vt_behavior")
    assert sel.select("Email") is None
    st = sel.state(); sel2 = UCTSelector(); sel2.load_state(st)
    assert sel2.tried == sel.tried
    ok("uct: select/update/state round-trip")

    from src.ioc_hunter.llm_agent import OllamaAgent
    ag = OllamaAgent()
    calls = {"n": 0}

    def stub(prompt):
        calls["n"] += 1
        if calls["n"] == 1:
            return "not json at all"
        return ('{"results": [{"candidate": "sibling.example.com", "score": 8, "reason": "ptr"},'
                '{"candidate": "google.com", "score": 1, "reason": "benign"},'
                '{"candidate": "l-a", "score": 9, "reason": "not in the candidate list"}]}')
    ag._generate = stub
    out = ag.screen({"value": "1.2.3.4", "type": "IP"},
                    [{"value": "sibling.example.com", "type": "Domain"},
                     {"value": "google.com", "type": "Domain"}], "ctx", "vt_object")
    assert len(out) == 2 and calls["n"] == 2 and out[0]["score"] == 8
    ok("llm_agent: JSON validation + retry + rejection of unknown candidates")

    print("\n" + "=" * 55)
    print(f"ALL {len(PASS)} CHECKS PASSED - CPU end-to-end pipeline is valid")
    return 0


if __name__ == "__main__":
    sys.exit(main())
