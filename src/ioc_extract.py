"""Stage 2 IOC extraction from raw text: ioc-finder plus extra regexes
(Registry, Port, Protocol, Filename, ...).

13 of the 19 ontology node types (Fig. 3) can be read straight out of the text:
  Domain, URL, IP, Email, Hash, Filename, CVE, Account, Port, Protocol, MAC, Registry, Header
(Technique/Tactic come from the TTP module; Time/Geo-location/CMD/API come from the
IOCHunter daemon in Stage 2.)

Main entry point:  extract_iocs(text) -> [{value, type, sent_idx}]
"""
import re

from .utils import sent_split

# refang the text before extracting
_REFANG = [
    (re.compile(r"\[\.\]|\(\.\)|\{\.\}"), "."),
    (re.compile(r"\[:\]"), ":"),
    (re.compile(r"\bhxxps?://", re.I), lambda m: m.group(0).lower().replace("hxxp", "http")),
    (re.compile(r"\[at\]|\(at\)", re.I), "@"),
]

RE_FILENAME = re.compile(
    r"\b[\w\-.%]{1,60}\.(?:exe|dll|sys|docx?|xlsx?|pptx?|pdf|zip|rar|7z|js|vbs|vbe|ps1|bat|cmd|lnk|scr|jar|apk|elf|sh|py|hta|iso|img|msi|tmp|dat|bin|txt|rtf|chm)\b",
    re.I)
RE_PORT = re.compile(r"\bports?\s+(\d{2,5})\b|(?<=[\w\]]):(\d{2,5})\b")
RE_PROTOCOL = re.compile(
    r"\b(HTTPS|HTTP|FTP|SFTP|SSH|RDP|SMB|SMTP|IMAP|POP3|DNS|TLS|SSL|ICMP|TCP|UDP|IRC|TELNET|SOCKS5?|WEBSOCKET)\b")
RE_ACCOUNT = re.compile(r"\b(?:user(?:name)?|account|login)\s*[:=]?\s*['\"]?([A-Za-z][\w.\-]{2,30})['\"]?", re.I)
RE_CVE = re.compile(r"\bCVE-\d{4}-\d{4,7}\b", re.I)
RE_MAC = re.compile(r"\b(?:[0-9A-Fa-f]{2}[:-]){5}[0-9A-Fa-f]{2}\b")
RE_REGISTRY = re.compile(r"\b(?:HKEY_[A-Z_]+|HKLM|HKCU|HKCR|HKU)\\[\w\\ .\-{}()]+", re.I)

_ACCOUNT_STOP = {"name", "names", "the", "this", "that", "your", "their", "data",
                 "information", "credentials", "password", "passwords"}


def refang(text: str) -> str:
    for pat, rep in _REFANG:
        text = pat.sub(rep, text)
    return text


def _finder(sent: str) -> list[tuple[str, str]]:
    """Use the ioc-finder library, falling back to plain regexes if it is missing."""
    out = []
    try:
        from ioc_finder import find_iocs
        d = find_iocs(sent)
        mapping = [
            ("domains", "Domain"), ("urls", "URL"), ("ipv4s", "IP"), ("ipv6s", "IP"),
            ("email_addresses", "Email"), ("email_addresses_complete", "Email"),
            ("md5s", "Hash"), ("sha1s", "Hash"), ("sha256s", "Hash"), ("sha512s", "Hash"),
            ("registry_key_paths", "Registry"), ("cves", "CVE"),
            ("mac_addresses", "MAC"), ("user_agents", "Header"),
        ]
        for key, typ in mapping:
            for v in d.get(key, []):
                out.append((v, typ))
    except ImportError:
        # minimal fallback (enough for tests; install ioc-finder for real runs)
        for pat, typ in [
            (r"\b(?:\d{1,3}\.){3}\d{1,3}\b", "IP"),
            (r"\bhttps?://[^\s\"'<>]+", "URL"),
            (r"\b[a-fA-F0-9]{64}\b|\b[a-fA-F0-9]{40}\b|\b[a-fA-F0-9]{32}\b", "Hash"),
            (r"\b[\w.\-]+@[\w.\-]+\.[a-z]{2,}\b", "Email"),
            (r"\b(?:[a-z0-9\-]+\.)+(?:com|net|org|io|ru|cn|info|biz|top|xyz|onion|gov|edu)\b", "Domain"),
        ]:
            for m in re.finditer(pat, sent, re.I):
                out.append((m.group(0), typ))
    return out


def _extra(sent: str) -> list[tuple[str, str]]:
    out = []
    for m in RE_FILENAME.finditer(sent):
        out.append((m.group(0), "Filename"))
    for m in RE_PORT.finditer(sent):
        p = m.group(1) or m.group(2)
        if p and 1 <= int(p) <= 65535:
            out.append((p, "Port"))
    for m in RE_PROTOCOL.finditer(sent):
        out.append((m.group(1).upper(), "Protocol"))
    for m in RE_ACCOUNT.finditer(sent):
        v = m.group(1)
        if v.lower() not in _ACCOUNT_STOP:
            out.append((v, "Account"))
    for m in RE_CVE.finditer(sent):
        out.append((m.group(0).upper(), "CVE"))
    for m in RE_MAC.finditer(sent):
        out.append((m.group(0), "MAC"))
    for m in RE_REGISTRY.finditer(sent):
        out.append((m.group(0), "Registry"))
    return out


def extract_iocs(text: str, sents: list[str] | None = None) -> list[dict]:
    """Extract IOCs sentence by sentence so each one carries a sent_idx for the CSKG.
    Deduplicated on (value, type), keeping the first occurrence."""
    if sents is None:
        sents = sent_split(text)
    seen, out = set(), []
    for i, sent in enumerate(sents):
        s = refang(sent)
        for value, typ in _finder(s) + _extra(s):
            value = value.strip().strip(".,;:'\"()[]")
            if not value:
                continue
            if typ == "Filename" and ("." not in value or len(value) < 5):
                continue
            key = (value.lower(), typ)
            if key in seen:
                continue
            seen.add(key)
            out.append({"value": value, "type": typ, "sent_idx": i})
    # a domain that also appears inside a URL/Email is kept: it is a distinct node type
    return out


if __name__ == "__main__":
    import json
    import sys
    text = sys.stdin.read() if not sys.stdin.isatty() else (
        "APT29 used spearphishing from evil[at]bad[.]com linking to hxxp://update-checker[.]net/a.exe "
        "(SHA256 e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855). "
        "The implant beacons to 45.77.12.9 on port 8443 over HTTPS, writes to "
        "HKLM\\Software\\Microsoft\\Windows\\CurrentVersion\\Run and exploits CVE-2023-23397.")
    print(json.dumps(extract_iocs(text), indent=1))
