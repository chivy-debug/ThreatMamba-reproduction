"""IOCHunter API clients: VirusTotal / AlienVault OTX / RapidDNS.

- VirusTotal free tier: 4 requests/minute -> RateLimiter at 15s/request, plus a 500/day
  quota tracked persistently by the runner.
- Every client returns a list of candidates: {value, type, attrs{date?, country?}, method}.
  vt_behavior additionally returns CMD/API nodes.
- HTTP 429 raises QuotaError so the runner can log it and move on instead of crashing.
"""
import base64
import os
import re
import time
from urllib.parse import quote, urlparse


class QuotaError(Exception):
    pass


class RateLimiter:
    def __init__(self, min_interval: float):
        self.min_interval = float(min_interval)
        self._last = 0.0

    def wait(self):
        dt = time.time() - self._last
        if dt < self.min_interval:
            time.sleep(self.min_interval - dt)
        self._last = time.time()


def _requests():
    import requests
    return requests


def _ts2date(ts) -> str | None:
    try:
        return time.strftime("%Y-%m-%d", time.gmtime(int(ts)))
    except (TypeError, ValueError):
        return None


class VTClient:
    BASE = "https://www.virustotal.com/api/v3"

    def __init__(self, key: str | None = None, sleep_seconds: float = 15):
        self.key = (key or os.getenv("VT_API_KEY", "")).strip()
        self.rl = RateLimiter(sleep_seconds)

    def _get(self, path: str) -> dict:
        """404 means "VirusTotal has never seen this object" -> return {} (NOT an error).
        Very common for URLs and hashes that were never submitted."""
        if not self.key:
            raise QuotaError("VT_API_KEY is empty")
        self.rl.wait()
        r = _requests().get(f"{self.BASE}{path}", headers={"x-apikey": self.key}, timeout=30)
        if r.status_code == 429:
            raise QuotaError("VT 429 rate-limit/quota")
        if r.status_code in (404, 400):
            return {}
        r.raise_for_status()
        return r.json()

    def object_info(self, ioc: dict) -> list[dict]:
        """vt_object: object details -> linked indicators + Geo-location + Time."""
        t, v = ioc["type"], ioc["value"]
        if t == "Hash":
            data = self._get(f"/files/{v}")
        elif t == "Domain":
            data = self._get(f"/domains/{v}")
        elif t == "IP":
            data = self._get(f"/ip_addresses/{v}")
        elif t == "URL":
            uid = base64.urlsafe_b64encode(v.encode()).decode().strip("=")
            data = self._get(f"/urls/{uid}")
        else:
            return []
        at = data.get("data", {}).get("attributes", {})
        out = []
        if at.get("country"):
            out.append({"value": at["country"], "type": "Geo-location", "attrs": {}, "method": "vt_object"})
        d = _ts2date(at.get("first_submission_date") or at.get("creation_date")
                     or at.get("last_modification_date"))
        if d:
            out.append({"value": d, "type": "Time", "attrs": {}, "method": "vt_object"})
        for rec in (at.get("last_dns_records") or [])[:10]:
            if rec.get("type") == "A" and rec.get("value"):
                out.append({"value": rec["value"], "type": "IP", "attrs": {}, "method": "vt_object"})
        for name in (at.get("names") or [])[:5]:
            if "." in name:
                out.append({"value": name, "type": "Filename", "attrs": {}, "method": "vt_object"})
        return out

    def behaviour(self, ioc: dict) -> list[dict]:
        """vt_behavior: sandbox report -> CMD + API nodes (hashes only)."""
        if ioc["type"] != "Hash":
            return []
        data = self._get(f"/files/{ioc['value']}/behaviour_summary")
        at = data.get("data", {})
        out = []
        for cmd in (at.get("command_executions") or [])[:10]:
            out.append({"value": str(cmd)[:200], "type": "CMD", "attrs": {}, "method": "vt_behavior"})
        for call in (at.get("calls_highlighted") or [])[:10]:
            out.append({"value": str(call)[:100], "type": "API", "attrs": {}, "method": "vt_behavior"})
        for f in (at.get("files_written") or [])[:5]:
            out.append({"value": str(f)[:150], "type": "Filename", "attrs": {}, "method": "vt_behavior"})
        return out


class OTXClient:
    BASE = "https://otx.alienvault.com/api/v1"
    SECTION = {"IP": "IPv4", "Domain": "domain", "URL": "url", "Hash": "file"}

    def __init__(self, key: str | None = None, timeout: float = 60, retries: int = 2):
        self.key = (key or os.getenv("OTX_API_KEY", "")).strip()
        self.rl = RateLimiter(2)
        self.timeout = timeout
        self.retries = retries

    def _get(self, path: str) -> dict:
        """Generous timeout plus retries: OTX is slow for indicators with many pulses.
        404 means no data -> return {}."""
        if not self.key:
            raise QuotaError("OTX_API_KEY is empty")
        last = None
        for attempt in range(self.retries + 1):
            self.rl.wait()
            try:
                r = _requests().get(f"{self.BASE}{path}",
                                    headers={"X-OTX-API-KEY": self.key}, timeout=self.timeout)
            except Exception as e:  # noqa: BLE001 - timeout / flaky network
                last = e
                continue
            if r.status_code == 429:
                raise QuotaError("OTX 429")
            if r.status_code == 404:
                return {}
            r.raise_for_status()
            return r.json()
        raise last if last else RuntimeError("OTX: unknown error")

    def general(self, ioc: dict) -> list[dict]:
        typ, val = ioc["type"], ioc["value"]
        # URLs: the OTX /indicators/url/... endpoint almost never returns Geo or passive DNS,
        # so looking the HOSTNAME up as a domain is far more productive
        if typ == "URL":
            host = (urlparse(val).hostname or "").strip(".")
            if not host:
                return []
            typ, val = "IP" if re.match(r"^\d+\.\d+\.\d+\.\d+$", host) else "Domain", host
        sec = self.SECTION.get(typ)
        if not sec:
            return []
        ioc = {"type": typ, "value": val}
        g = self._get(f"/indicators/{sec}/{quote(val, safe='')}/general")
        if not g:
            return []
        out = []
        if g.get("country_name") or g.get("country_code"):
            out.append({"value": g.get("country_name") or g["country_code"],
                        "type": "Geo-location", "attrs": {}, "method": "otx_general"})
        # passive DNS: linked domains/IPs + Time
        if sec in ("IPv4", "domain"):
            try:
                pdns = self._get(f"/indicators/{sec}/{quote(ioc['value'], safe='')}/passive_dns")
                for rec in (pdns.get("passive_dns") or [])[:15]:
                    host, addr = rec.get("hostname"), rec.get("address")
                    if host and host != ioc["value"]:
                        out.append({"value": host, "type": "Domain", "attrs": {}, "method": "otx_general"})
                    if addr and addr != ioc["value"] and re.match(r"^\d+\.\d+\.\d+\.\d+$", str(addr)):
                        out.append({"value": addr, "type": "IP", "attrs": {}, "method": "otx_general"})
                    d = str(rec.get("first") or "")[:10]
                    if d:
                        out.append({"value": d, "type": "Time", "attrs": {}, "method": "otx_general"})
            except QuotaError:
                raise
            except Exception:  # noqa: BLE001 - passive_dns is a nice-to-have
                pass
        return out


class RapidDNSClient:
    """The real rapiddns.io results table has 5 columns:

        # | Domain | Address | Type | Date

    An earlier regex assumed 4 columns and bare <td> tags (no attributes), so it never
    matched: 1,219 successful calls produced 0 candidates. This version splits on
    <tr>...</tr> and then on each <td> (accepting any attributes), stripping inner HTML.
    """
    BASE = "https://rapiddns.io"
    _TR = re.compile(r"<tr[^>]*>(.*?)</tr>", re.S | re.I)
    _TD = re.compile(r"<td[^>]*>(.*?)</td>", re.S | re.I)
    _TAG = re.compile(r"<[^>]+>")
    _DATE = re.compile(r"^\d{4}-\d{2}-\d{2}")
    _IPV4 = re.compile(r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$")

    def __init__(self, timeout: float = 30):
        self.rl = RateLimiter(3)
        self.timeout = timeout

    @classmethod
    def _cell(cls, html: str) -> str:
        import html as _h
        return _h.unescape(cls._TAG.sub("", html)).strip()

    @classmethod
    def parse_rows(cls, html: str) -> list[list[str]]:
        """Return the cleaned rows, keeping only those with at least 4 cells."""
        rows = []
        for tr in cls._TR.findall(html):
            cells = [cls._cell(td) for td in cls._TD.findall(tr)]
            if len(cells) >= 4:
                rows.append(cells)
        return rows

    def history(self, ioc: dict) -> list[dict]:
        """rapiddns_history: Domain -> subdomains; IP -> same-IP hosts. Time comes from
        the Date column."""
        self.rl.wait()
        path = f"/subdomain/{ioc['value']}" if ioc["type"] == "Domain" else f"/sameip/{ioc['value']}"
        r = _requests().get(f"{self.BASE}{path}?full=1",
                            headers={"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) "
                                                   "AppleWebKit/537.36 (KHTML, like Gecko) "
                                                   "Chrome/124.0 Safari/537.36"},
                            timeout=self.timeout)
        if r.status_code == 429:
            raise QuotaError("RapidDNS 429")
        r.raise_for_status()

        out, seen = [], set()
        self_val = ioc["value"].lower()
        for cells in self.parse_rows(r.text):
            # drop the row-number column when present (first cell is an integer)
            if cells[0].isdigit() and len(cells) >= 5:
                cells = cells[1:]
            dom, addr, rtype = cells[0], cells[1], cells[2]
            date = next((c for c in reversed(cells) if self._DATE.match(c)), "")
            for cand in (dom, addr):
                key = cand.lower()
                if not cand or key == self_val or key in seen or "." not in cand or " " in cand:
                    continue
                seen.add(key)
                out.append({"value": cand,
                            "type": "IP" if self._IPV4.match(cand) else "Domain",
                            "attrs": {"record": rtype}, "method": "rapiddns_history"})
            if date and date[:10] not in seen:
                seen.add(date[:10])
                out.append({"value": date[:10], "type": "Time", "attrs": {},
                            "method": "rapiddns_history"})
            if len(out) >= 40:
                break
        return out


def call_method(method: str, ioc: dict, vt: VTClient, otx: OTXClient, rdns: RapidDNSClient) -> list[dict]:
    if method == "vt_object":
        return vt.object_info(ioc)
    if method == "vt_behavior":
        return vt.behaviour(ioc)
    if method == "otx_general":
        return otx.general(ioc)
    if method == "rapiddns_history":
        return rdns.history(ioc)
    raise ValueError(f"unknown method: {method}")
