#!/usr/bin/env python3
"""Stage 0 check: issue one real request against each API - VirusTotal, AlienVault OTX,
RapidDNS.

Usage:  python scripts/checks/check_apis.py {vt|otx|rapiddns}
Keys are read from the VT_API_KEY / OTX_API_KEY environment variables. smoke_test.sh loads
.env itself; when run standalone this script loads .env via python-dotenv if available.

Exit codes: 0 = PASS, 1 = FAIL, 2 = SKIP (SMOKE_OFFLINE=1).
"""
import os
import sys


def result(name: str, ok: bool, msg: str, skip: bool = False) -> int:
    tag = "SKIP" if skip else ("PASS" if ok else "FAIL")
    print(f"[{tag}] {name}: {msg}")
    return 2 if skip else (0 if ok else 1)


def _requests():
    try:
        import requests
        return requests
    except ImportError:
        return None


def check_vt() -> int:
    if os.getenv("SMOKE_OFFLINE"):
        return result("vt", False, "SMOKE_OFFLINE=1", skip=True)
    key = os.getenv("VT_API_KEY", "").strip()
    if not key:
        return result("vt", False,
                      "VT_API_KEY is not set. Sign up free at virustotal.com -> avatar -> "
                      "API key, then put it in .env")
    requests = _requests()
    if requests is None:
        return result("vt", False, "'requests' is missing -> pip install -r requirements.txt")
    try:
        r = requests.get("https://www.virustotal.com/api/v3/ip_addresses/8.8.8.8",
                         headers={"x-apikey": key}, timeout=20)
    except Exception as e:
        return result("vt", False, f"network error: {e}")
    if r.status_code == 200:
        return result("vt", True, "GET /api/v3/ip_addresses/8.8.8.8 -> 200 OK")
    if r.status_code == 401:
        return result("vt", False, "401 - invalid key, or the account is not activated")
    if r.status_code == 429:
        return result("vt", True, "429 - valid key but currently rate-limited "
                                  "(acceptable for the smoke test)")
    return result("vt", False, f"HTTP {r.status_code}: {r.text[:150]}")


def check_otx() -> int:
    if os.getenv("SMOKE_OFFLINE"):
        return result("otx", False, "SMOKE_OFFLINE=1", skip=True)
    key = os.getenv("OTX_API_KEY", "").strip()
    if not key:
        return result("otx", False,
                      "OTX_API_KEY is not set. Sign up free at otx.alienvault.com -> "
                      "Settings -> OTX Key, then put it in .env")
    requests = _requests()
    if requests is None:
        return result("otx", False, "'requests' is missing -> pip install -r requirements.txt")
    try:
        r = requests.get("https://otx.alienvault.com/api/v1/indicators/IPv4/8.8.8.8/general",
                         headers={"X-OTX-API-KEY": key}, timeout=20)
    except Exception as e:
        return result("otx", False, f"network error: {e}")
    if r.status_code == 200:
        return result("otx", True, "GET /api/v1/indicators/IPv4/8.8.8.8/general -> 200 OK")
    if r.status_code in (401, 403):
        return result("otx", False, f"{r.status_code} - invalid key, or the email address "
                                    f"is not confirmed")
    return result("otx", False, f"HTTP {r.status_code}: {r.text[:150]}")


def check_rapiddns() -> int:
    if os.getenv("SMOKE_OFFLINE"):
        return result("rapiddns", False, "SMOKE_OFFLINE=1", skip=True)
    requests = _requests()
    if requests is None:
        return result("rapiddns", False,
                      "'requests' is missing -> pip install -r requirements.txt")
    try:
        r = requests.get("https://rapiddns.io/sameip/8.8.8.8",
                         headers={"User-Agent": "Mozilla/5.0 (X11; Linux x86_64)"}, timeout=25)
    except Exception as e:
        return result("rapiddns", False, f"network error: {e}")
    if r.status_code == 200 and len(r.text) > 500:
        return result("rapiddns", True,
                      "GET /sameip/8.8.8.8 -> 200 with content (no API key needed)")
    return result("rapiddns", False,
                  f"HTTP {r.status_code}, len={len(r.text)} - possibly blocked temporarily "
                  f"by Cloudflare; try again later")


def main() -> int:
    which = sys.argv[1] if len(sys.argv) > 1 else ""
    fn = {"vt": check_vt, "otx": check_otx, "rapiddns": check_rapiddns}.get(which)
    if fn is None:
        print("Usage: python scripts/checks/check_apis.py {vt|otx|rapiddns}")
        return 1
    try:
        from dotenv import load_dotenv
        load_dotenv()  # load .env when running standalone
    except ImportError:
        pass
    return fn()


if __name__ == "__main__":
    sys.exit(main())
