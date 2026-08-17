"""IOCHunter-LLM: the agent that selects (IOC, hunting method) pairs and screens candidates.

TWO FUNCTIONS WITH TWO DIFFERENT ROLES - read this before editing:

1. `select_pairs()` - FAITHFUL TO THE PAPER (Fig. 2 + Algorithm 1, lines 22-23).
   The LLM receives "Screening Objects" (IOCs with Context/Coverage/Tried Time) and
   "Hunting Methods" (methods with Description/Coverage/Tried Time), and returns
   {IOC, Method, Score 0-100, Reason} pairs. This is the PLANNING step: deciding which API
   to call for which IOC BEFORE spending a request. In Algorithm 1:
       21: M_o    <- Top_{k=50}(M_order)          # UCT ranking
       23: M'_o   <- IOCHunter-LLM(M_o, o)        # LLM refinement
   The paper uses DeepSeek-R1-70B; this reproduction uses Qwen3-8B, which fits in 16 GB VRAM.

2. `screen()` - AN EXTENSION BEYOND THE PAPER, optional.
   Scores 0-10 the candidates the API HAS ALREADY RETURNED, filtering out noise. The paper
   does not describe this step (every hunting result goes into the CSKG). It is kept here
   because this CTI corpus contains many benign domains, and is toggled via
   `iochunter.post_screen` in the config.

Both validate the JSON response and retry up to `max_retries` times.
"""
import json
import os

# ------------------------------------------------------- Fig. 2 (verbatim from the paper)
SELECT_TMPL = """Task Explanation: You are an outstanding cybersecurity analyst. You are asked to \
rate the following filters based on your cyber threat hunting experience to select suitable \
hunting nodes and methods. We will give a list and description of the screening objects.

Screening Objects: List<Object>
{objects}

Hunting Methods: List<Object>
{methods}

Output Format: List<Object>
Return ONLY a JSON object of the form:
{{"results": [{{"IOC": "<the IOC value, copied exactly>", "Method": "<one method name from \
the list above>", "Score": <integer 0-100>, "Reason": "<one short sentence>"}}]}}

Rules:
- Score 0-100 expresses how likely this (IOC, Method) pair will reveal NEW attack \
infrastructure. Higher is better.
- Only emit pairs whose Method is applicable to that IOC's type.
- Prefer pairs with low "Tried Time" but plausible payoff; avoid wasting the quota on \
IOCs that look like benign or shared infrastructure.
- Emit at most {max_pairs} pairs, best first.
"""

# ------------------------------------------ extension: screen the returned candidates
SCREEN_TMPL = """You are an IOC screening agent in a cyber threat intelligence hunting pipeline.

## Screening Objects
Seed IOC: {seed_value} (type: {seed_type})
Context from the CTI report:
\"\"\"{context}\"\"\"

## Hunting Method
Candidates below were returned by the hunting function `{method}` applied to the seed IOC.

## Candidates
{cand_block}

## Task
For EACH candidate, rate 0-10 how likely it is REAL attack infrastructure/artifact related to
the seed IOC and the context (10 = certainly related & malicious, 0 = unrelated/benign/noise).
Penalize: CDN/shared hosting, parking pages, popular benign domains, sinkholes, private/reserved IPs.

## Output Format
Respond with ONLY a JSON object:
{{"results": [{{"candidate": "<value>", "score": <int 0-10>, "reason": "<short>"}}, ...]}}
"""


class OllamaAgent:
    def __init__(self, host: str | None = None, model: str | None = None,
                 max_retries: int = 3, timeout: int = 300):
        self.host = (host or os.getenv("OLLAMA_HOST", "http://localhost:11434")).rstrip("/")
        self.model = model or os.getenv("OLLAMA_MODEL", "qwen3:8b")
        self.max_retries = max_retries
        self.timeout = timeout

    def _generate(self, prompt: str) -> str:
        import requests
        payload = {"model": self.model, "prompt": prompt, "stream": False,
                   "format": "json", "options": {"temperature": 0}, "think": False}
        r = requests.post(f"{self.host}/api/generate", json=payload, timeout=self.timeout)
        if r.status_code == 400 and "think" in payload:
            payload.pop("think")
            r = requests.post(f"{self.host}/api/generate", json=payload, timeout=self.timeout)
        r.raise_for_status()
        return r.json().get("response", "")

    def _json_retry(self, prompt: str, parse):
        """Call the LLM at most max_retries times, running `parse(obj)` on each response.
        A parse returning None counts as a failure and triggers a retry."""
        last = ""
        for _ in range(self.max_retries):
            try:
                raw = self._generate(prompt)
                out = parse(json.loads(raw))
                if out is not None:
                    return out
                last = f"parse tra None: {raw[:150]}"
            except Exception as e:  # noqa: BLE001
                last = f"{type(e).__name__}: {e}"
        raise RuntimeError(f"LLM failed after {self.max_retries} attempts: {last}")

    # ------------------------------------------------------------- PAPER: Fig. 2
    def select_pairs(self, objects: list[dict], methods: list[dict],
                     max_pairs: int = 8) -> list[dict]:
        """Algorithm 1, line 23: choose (IOC, Method) pairs before spending API calls.

        objects: [{"IOC":..., "Type":..., "Context":..., "Coverage":int, "Tried Time":int}]
        methods: [{"Method":..., "Source":..., "Description":..., "Coverage":int, "Tried Time":int}]
        Returns: [{"ioc": <value>, "method": <name>, "score": float 0-100, "reason": str}]
                 with invalid pairs dropped, sorted by descending score.
        """
        if not objects or not methods:
            return []
        prompt = SELECT_TMPL.format(
            objects="\n".join(json.dumps(o, ensure_ascii=False) for o in objects),
            methods="\n".join(json.dumps(m, ensure_ascii=False) for m in methods),
            max_pairs=max_pairs)
        by_ioc = {str(o["IOC"]).lower(): o for o in objects}
        valid_m = {m["Method"] for m in methods}

        def parse(obj):
            res = obj.get("results")
            if not isinstance(res, list):
                return None
            out = []
            for r in res:
                v = str(r.get("IOC", "")).strip()
                m = str(r.get("Method", "")).strip()
                s = r.get("Score")
                if v.lower() not in by_ioc or m not in valid_m:
                    continue
                if not isinstance(s, (int, float)) or isinstance(s, bool):
                    continue
                out.append({"ioc": by_ioc[v.lower()]["IOC"], "method": m,
                            "score": float(s), "reason": str(r.get("Reason", ""))[:300]})
            # res == [] is a valid answer ("no pair is worth trying")
            return sorted(out, key=lambda d: -d["score"])[:max_pairs] if (out or res == []) else None

        return self._json_retry(prompt, parse)

    # ------------------------------------------------- extension beyond the paper
    def screen(self, seed: dict, candidates: list[dict], context: str, method: str) -> list[dict]:
        """candidates: [{value,type,...}] -> [{value,type,score,reason}], valid candidates only."""
        if not candidates:
            return []
        cand_block = "\n".join(f"- {c['value']} (type: {c['type']})" for c in candidates[:30])
        prompt = SCREEN_TMPL.format(seed_value=seed["value"], seed_type=seed["type"],
                                    context=context[:1500], method=method, cand_block=cand_block)
        by_val = {c["value"].lower(): c for c in candidates}

        def parse(obj):
            res = obj.get("results")
            if not isinstance(res, list):
                return None
            out = []
            for r in res:
                v = str(r.get("candidate", "")).strip()
                s = r.get("score")
                if not isinstance(s, (int, float)) or isinstance(s, bool):
                    continue
                c = by_val.get(v.lower())
                if c is None:
                    continue
                out.append({**c, "score": float(s), "reason": str(r.get("reason", ""))[:300]})
            return out if (out or res == []) else None

        return self._json_retry(prompt, parse)
