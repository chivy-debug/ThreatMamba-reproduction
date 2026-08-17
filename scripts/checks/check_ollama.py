#!/usr/bin/env python3
"""Stage 0 check: Ollama + qwen3:8b return well-formed JSON (validated, up to 3 retries).

The prompt mirrors the IOCHunter output structure (Fig. 2: Score + Reason) so this tests
exactly what Stage 2 will need.

Exit codes: 0 = PASS, 1 = FAIL, 2 = SKIP (SMOKE_OFFLINE=1).
Run standalone:  python scripts/checks/check_ollama.py
"""
import json
import os
import sys

PROMPT = (
    "You are an IOC screening assistant for cyber threat intelligence.\n"
    'Context: "The malware beacons to update-checker[.]net every 60 seconds."\n'
    "Candidate indicator: update-checker.net\n"
    "Rate how likely this candidate is malicious infrastructure related to the context.\n"
    'Respond with ONLY a JSON object: {"score": <integer 0-10>, "reason": "<one short sentence>"}'
)


def fail(msg: str) -> int:
    print(f"[FAIL] ollama: {msg}")
    return 1


def main() -> int:
    if os.getenv("SMOKE_OFFLINE"):
        print("[SKIP] ollama: SMOKE_OFFLINE=1")
        return 2

    try:
        import requests
    except ImportError:
        return fail("'requests' is missing -> pip install -r requirements.txt")

    host = os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
    model = os.getenv("OLLAMA_MODEL", "qwen3:8b")

    # 1) Is the server up? Has the model been pulled?
    try:
        r = requests.get(f"{host}/api/tags", timeout=10)
        r.raise_for_status()
    except Exception as e:
        return fail(
            f"cannot connect to {host} ({e}).\n"
            "       Is Ollama running? Try 'ollama serve' (foreground) or "
            "'sudo systemctl start ollama'."
        )
    names = [m.get("name", "") for m in r.json().get("models", [])]
    if model not in names:
        return fail(f"model '{model}' is not available (present: {names or '[none]'}). "
                    f"Run: ollama pull {model}")

    # 2) Generate structured JSON, retrying up to 3 times
    payload = {
        "model": model,
        "prompt": PROMPT,
        "stream": False,
        "format": "json",           # force Ollama to return JSON
        "options": {"temperature": 0},
        "think": False,             # disable qwen3 thinking mode (Ollama >= 0.9)
    }
    last_err = ""
    for attempt in range(1, 4):
        try:
            r = requests.post(f"{host}/api/generate", json=payload, timeout=300)
            if r.status_code == 400 and "think" in payload:
                payload.pop("think")  # older Ollama builds reject the 'think' parameter
                continue
            r.raise_for_status()
            raw = r.json().get("response", "")
            obj = json.loads(raw)
            score = obj.get("score")
            reason = obj.get("reason")
            score_ok = isinstance(score, (int, float)) and not isinstance(score, bool) and 0 <= score <= 10
            if score_ok and isinstance(reason, str) and reason.strip():
                print(f"[PASS] ollama: {model} returned valid JSON after {attempt} "
                      f"attempt(s) (score={score})")
                return 0
            last_err = f"JSON missing or malformed keys: {raw[:200]!r}"
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
    return fail(
        f"three attempts produced no valid JSON. Last error: {last_err}\n"
        "       What to try: update Ollama to the latest build; if it still misbehaves, "
        "add few-shot\n"
        "       examples to the prompt; failing that, swap in another model of "
        "similar size."
    )


if __name__ == "__main__":
    sys.exit(main())
