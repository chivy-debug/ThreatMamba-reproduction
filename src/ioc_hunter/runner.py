"""IOCHunter background daemon (Algorithm 1): resumable, checkpointed after every IOC.

Run:    python -m src.ioc_hunter.runner [--once N] [--docs path]
Stop:   Ctrl+C (SIGINT) -> saves state and exits cleanly.

HARD RULE: enriched results are used ONLY for the demo/UI. They are NEVER mixed into the
classifier's training set.

Output: data/enriched/{doc_id}.json
  {doc_id, group, iocs: [...],
   hunts: [{seed, method, candidates_raw, accepted: [{value, type, score, reason}]}]}
State:  data/enriched/_state.json  {done_docs, ioc_cache, uct, vt_calls: {date: n}}
Log:    outputs/enrich.log (one event per line; ERROR lines carry a short traceback)
"""
import argparse
import datetime as dt
import json
import os
import signal
import sys

from ..ioc_extract import extract_iocs
from ..utils import ENRICHED, OUTPUTS, PROCESSED, ROOT, load_config, read_jsonl, sent_split
from .apis import OTXClient, QuotaError, RapidDNSClient, VTClient, call_method
from .filters import select_seeds
from .llm_agent import OllamaAgent
from .uct import UCTSelector


def load_env_or_die():
    """Load .env and abort immediately if a key is missing.

    Historical bug: run_enrichment_daemon.sh did not load .env and the runner did not load
    it either, so VT/OTX ran with empty keys, every call raised QuotaError, and the daemon
    "ran" for four hours collecting nothing. This now fails in one second."""
    env_fp = ROOT / ".env"
    if env_fp.exists():
        try:
            from dotenv import load_dotenv
            load_dotenv(env_fp, override=False)
        except ImportError:      # python-dotenv missing -> parse it by hand
            for line in env_fp.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
    missing = [k for k in ("VT_API_KEY", "OTX_API_KEY") if not os.getenv(k, "").strip()]
    if missing:
        raise SystemExit(
            f"MISSING API KEY(S): {', '.join(missing)}\n"
            f"   Check {env_fp} (copy .env.example and fill in your keys).\n"
            "   Without keys, VT/OTX fail 100% of the time - do not waste hours running.")


STATE_FP = ENRICHED / "_state.json"
LOG_FP = OUTPUTS / "enrich.log"
HUNTABLE = ("Domain", "IP", "URL", "Hash")


def log(msg: str):
    LOG_FP.parent.mkdir(parents=True, exist_ok=True)
    line = f"{dt.datetime.now().isoformat(timespec='seconds')} {msg}"
    print(line, flush=True)
    with open(LOG_FP, "a", encoding="utf-8") as f:
        f.write(line + "\n")


class Daemon:
    def __init__(self, cfg):
        icfg = cfg["iochunter"]
        self.cfg = cfg
        self.max_rounds = int(icfg.get("max_rounds", 5))
        self.max_seeds = int(icfg.get("max_seeds_per_doc", 12))
        self.accept_score = float(icfg.get("accept_score", 6))
        self.top_k_methods = int(icfg.get("top_k_methods", 50))
        self.select_score_min = float(icfg.get("select_score_min", 50))
        self.post_screen = bool(icfg.get("post_screen", True))
        self.vt_daily_quota = int(icfg.get("vt_daily_quota", 480))
        self.vt = VTClient(sleep_seconds=float(icfg.get("vt_sleep_seconds", 15)))
        self.otx = OTXClient()
        self.rdns = RapidDNSClient()
        self.llm = OllamaAgent(model=icfg.get("llm"), max_retries=int(icfg.get("max_llm_retries", 3)))
        self.uct = UCTSelector()
        self.state = {"done_docs": [], "ioc_cache": {}, "uct": {}, "vt_calls": {},
                      "ioc_stats": {}}
        self._stop = False
        if STATE_FP.exists():
            self.state = json.load(open(STATE_FP, encoding="utf-8"))
            self.uct.load_state(self.state.get("uct", {}))
            log(f"resume: {len(self.state['done_docs'])} documents done, "
                f"{len(self.state['ioc_cache'])} IOCs cached")

    # ---------------- state
    def save_state(self):
        self.state["uct"] = self.uct.state()
        ENRICHED.mkdir(parents=True, exist_ok=True)
        tmp = STATE_FP.with_suffix(".tmp")
        json.dump(self.state, open(tmp, "w", encoding="utf-8"))
        tmp.replace(STATE_FP)

    def vt_budget_left(self) -> int:
        today = dt.date.today().isoformat()
        return self.vt_daily_quota - int(self.state["vt_calls"].get(today, 0))

    def count_vt(self):
        today = dt.date.today().isoformat()
        self.state["vt_calls"][today] = int(self.state["vt_calls"].get(today, 0)) + 1

    # ---------------- per-IOC counters (the Coverage / Tried Time columns of Fig. 2)
    def ioc_key(self, ioc: dict) -> str:
        return f"{ioc['type']}::{ioc['value'].lower()}"

    def ioc_stat(self, ioc: dict) -> dict:
        return self.state.setdefault("ioc_stats", {}).setdefault(
            self.ioc_key(ioc), {"tried": 0, "coverage": 0})

    # ------------- Algorithm 1, lines 18-23: UCT ranking -> LLM refinement
    def plan_pairs(self, seeds: list[dict], ctx_of, tried: set[tuple[str, str]]) -> list[tuple]:
        """Return a priority-ordered list of (ioc, method) pairs.

        Follows the paper exactly:
          18-21: rank methods by s_{m,o} (Eq. 1) and take the top k=50
          22-23: hand both the IOC list and the method list to IOCHunter-LLM for refinement
        If the LLM fails, fall back to pure UCT: the daemon must never stop because of the
        language model."""
        pending = [s for s in seeds
                   if any((self.ioc_key(s), m) not in tried
                          for m in self.uct.applicable(s["type"]))]
        if not pending:
            return []
        method_names, seen = [], set()
        for s in pending:
            for m in self.uct.rank(s["type"], top_k=self.top_k_methods):
                if m not in seen:
                    seen.add(m); method_names.append(m)
        objects = []
        for s in pending:
            st = self.ioc_stat(s)
            objects.append({"IOC": s["value"], "Type": s["type"],
                            "Context": ctx_of(s)[:400],
                            "Coverage": st["coverage"], "Tried Time": st["tried"]})
        by_val = {s["value"].lower(): s for s in pending}
        try:
            pairs = self.llm.select_pairs(objects, [self.uct.describe(m) for m in method_names],
                                          max_pairs=self.max_rounds * 2)
            out = []
            for p in pairs:
                s = by_val.get(str(p["ioc"]).lower())
                if s is None or p["score"] < self.select_score_min:
                    continue
                if (self.ioc_key(s), p["method"]) in tried:
                    continue
                if p["method"] not in self.uct.applicable(s["type"]):
                    continue
                out.append((s, p["method"], p["score"], p["reason"]))
            if out:
                log(f"  LLM selected {len(out)} (IOC, method) pairs: "
                    + ", ".join(f"{m}<-{s['value'][:24]}({sc:.0f})" for s, m, sc, _ in out[:4]))
                return out
            log("  LLM selected no pairs -> falling back to plain UCT")
        except Exception as e:  # noqa: BLE001
            log(f"  LLM-SELECT-FAIL: {type(e).__name__}: {e} -> falling back to plain UCT")
        # fallback: let UCT pick the best method for each seed
        out = []
        for s in pending:
            m = self.uct.select(s["type"],
                                exclude={mm for (k, mm) in tried if k == self.ioc_key(s)})
            if m:
                out.append((s, m, 0.0, "UCT fallback"))
        return out

    # ---------------- execute a single (IOC, method) pair
    def hunt_pair(self, ioc: dict, method: str, context: str) -> dict | None:
        if method.startswith("vt_") and self.vt_budget_left() <= 0:
            log(f"  SKIP {method} (VirusTotal daily quota exhausted)")
            return None
        try:
            if method.startswith("vt_"):
                self.count_vt()
            cands = call_method(method, ioc, self.vt, self.otx, self.rdns)
        except QuotaError as e:
            log(f"  QUOTA {method} {ioc['value']}: {e}")
            self.uct.update(method, 0); return None
        except Exception as e:  # noqa: BLE001
            log(f"  ERROR {method} {ioc['value']}: {type(e).__name__}: {e}")
            self.uct.update(method, 0); return None
        # Time/Geo/CMD/API are authoritative attributes from the source, so they are accepted
        # directly. Linked IOCs may go through the extra screening step (post_screen, which
        # is an extension beyond the paper).
        direct = [c for c in cands if c["type"] in ("Time", "Geo-location", "CMD", "API")]
        rest = [c for c in cands if c not in direct]
        accepted = list(direct)
        if rest and self.post_screen:
            try:
                scored = self.llm.screen(ioc, rest, context, method)
                accepted += [c for c in scored if c["score"] >= self.accept_score]
            except Exception as e:  # noqa: BLE001
                log(f"  LLM-SCREEN-FAIL {method}: {e} -> accepting all candidates this round")
                accepted += rest
        elif rest:
            accepted += rest
        self.uct.update(method, len(accepted))
        st = self.ioc_stat(ioc)
        st["tried"] += 1; st["coverage"] += len(accepted)
        log(f"  {method}: {len(cands)} cand -> {len(accepted)} accepted "
            f"(UCT {self.uct.tried[method]}/{self.uct.coverage[method]})")
        return {"seed": ioc, "method": method, "n_candidates": len(cands), "accepted": accepted}

    # ---------------- one document (Algorithm 1, while loop lines 5-35)
    def process_doc(self, doc: dict):
        doc_id = doc["doc_id"]
        sents = sent_split(doc["text"])
        iocs = extract_iocs(doc["text"], sents)
        # SEED FILTERING: drop security-vendor/news/government domains, private IPs, ...;
        # priority Hash > defanged IOC > IP > remaining domains (see filters.py)
        seeds = select_seeds(iocs, doc["text"], self.max_seeds)
        log(f"DOC {doc_id} ({doc['group']}): {len(iocs)} IOCs -> {len(seeds)} seeds worth hunting "
            f"({', '.join(sorted({s['type'] for s in seeds})) or 'none'})")
        result = {"doc_id": doc_id, "group": doc["group"], "iocs": iocs, "hunts": []}
        out_fp = ENRICHED / f"{doc_id}.json"

        def ctx_of(s):
            return sents[s["sent_idx"]] if s["sent_idx"] < len(sents) else doc["text"][:1000]

        # reuse results already hunted for this IOC in another document (line 4: O_used)
        tried: set[tuple[str, str]] = set()
        for s in seeds:
            for h in self.state["ioc_cache"].get(self.ioc_key(s), []):
                result["hunts"].append(h)
                tried.add((self.ioc_key(s), h["method"]))

        for _round in range(self.max_rounds):
            if self._stop:
                break
            plan = self.plan_pairs(seeds, ctx_of, tried)
            if not plan:
                break
            for ioc, method, _score, _reason in plan:
                if self._stop:
                    break
                tried.add((self.ioc_key(ioc), method))
                h = self.hunt_pair(ioc, method, ctx_of(ioc))
                if h:
                    result["hunts"].append(h)
                    self.state["ioc_cache"].setdefault(self.ioc_key(ioc), []).append(h)
                ENRICHED.mkdir(parents=True, exist_ok=True)   # checkpoint after every call
                json.dump(result, open(out_fp, "w", encoding="utf-8"), ensure_ascii=False)
                self.save_state()
        if self._stop:
            return
        n_new = sum(len(h["accepted"]) for h in result["hunts"])
        # Only mark a document "done" when it actually yielded nodes. If the network or an
        # API was broken, keep it queued for a later run (historical bug: 155 documents were
        # silently lost this way).
        if n_new > 0 or not seeds:
            self.state["done_docs"].append(doc_id)
            self.save_state()
            log(f"DONE {doc_id}: {len(result['hunts'])} hunts, {n_new} new nodes")
        else:
            self.state.setdefault("failed_docs", {})[doc_id] = \
                int(self.state.get("failed_docs", {}).get(doc_id, 0)) + 1
            if self.state["failed_docs"][doc_id] >= 3:
                self.state["done_docs"].append(doc_id)   # give up after three empty attempts
                log(f"GIVEUP {doc_id}: three attempts yielded no nodes")
            else:
                log(f"RETRY-LATER {doc_id}: 0 new nodes "
                    f"(attempt {self.state['failed_docs'][doc_id]}/3)")
            self.save_state()

    # ---------------- main loop
    def run(self, docs: list[dict], once: int | None):
        signal.signal(signal.SIGINT, self._sig)
        signal.signal(signal.SIGTERM, self._sig)
        todo = [d for d in docs if d["doc_id"] not in set(self.state["done_docs"])]
        log(f"daemon start: {len(todo)}/{len(docs)} documents to process | "
            f"VirusTotal budget left today: {self.vt_budget_left()}")
        n = 0
        for doc in todo:
            if self._stop:
                break
            self.process_doc(doc)
            n += 1
            if once and n >= once:
                log(f"--once {once}: stopping as requested")
                break
        self.save_state()
        log("daemon exited cleanly")

    def _sig(self, *_a):
        log("received SIGINT/SIGTERM -> saving state...")
        self._stop = True


def load_queue(docs_path: str | None) -> list[dict]:
    """Work queue: demo_subset.txt first, then the rest of the training split."""
    all_docs = {d["doc_id"]: d for d in read_jsonl(PROCESSED / "docs.jsonl")}
    order: list[str] = []
    subset_fp = PROCESSED.parent / "demo_subset.txt" if docs_path is None else None
    if docs_path:
        order = [l.strip() for l in open(docs_path, encoding="utf-8") if l.strip()]
    elif subset_fp and subset_fp.exists():
        order = [l.strip() for l in subset_fp.read_text(encoding="utf-8").splitlines() if l.strip()]
    rest = [did for did, d in sorted(all_docs.items()) if d["split"] == "train" and did not in set(order)]
    return [all_docs[i] for i in order + rest if i in all_docs]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", type=int, default=None,
                    help="process N documents then exit")
    ap.add_argument("--docs", default=None,
                    help="file with a list of doc_ids, used instead of demo_subset.txt")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    load_env_or_die()
    cfg = load_config(args.config)
    daemon = Daemon(cfg)
    daemon.run(load_queue(args.docs), args.once)


if __name__ == "__main__":
    sys.exit(main())
