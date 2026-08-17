"""Filter and prioritise SEED IOCs before spending API calls (Stage 2).

WHY THIS EXISTS:
  ioc-finder cannot distinguish "this domain is an IOC" from "this domain is cited in the
  article". In practice the daemon burned VirusTotal quota looking up pwc.com,
  whitehouse.gov, eset.com, teamviewer.com and similar - all reference sources, not attack
  infrastructure. The paper uses an LLM to screen the RETURNED candidates; this module
  screens the INPUT seeds.

RULES
  - Always drop: domains/URLs belonging to security vendors, news outlets, government
    agencies, big tech and CDNs.
  - Always drop: private/reserved/loopback/documentation IPs.
  - Priority: Hash > IOC that was defanged in the source text > IP > any other Domain/URL.
    (A defanged form such as `evil[.]com` is very strong evidence the author treated it
    as an IOC.)
"""
import ipaddress
import re
from urllib.parse import urlparse

# --- benign domains (matched by registered suffix, e.g. "news.eset.com" -> "eset.com") ---
BENIGN_DOMAINS = {
    # security vendors / threat intel
    "eset.com", "welivesecurity.com", "kaspersky.com", "kaspersky.ru", "securelist.com",
    "symantec.com", "broadcom.com", "mcafee.com", "trendmicro.com", "sophos.com",
    "crowdstrike.com", "fireeye.com", "mandiant.com", "paloaltonetworks.com", "unit42.com",
    "checkpoint.com", "research.checkpoint.com", "fortinet.com", "malwarebytes.com",
    "avast.com", "avg.com", "bitdefender.com", "f-secure.com", "withsecure.com",
    "drweb.com", "virustotal.com", "hybrid-analysis.com", "any.run", "joesandbox.com",
    "talosintelligence.com", "secureworks.com", "proofpoint.com", "recordedfuture.com",
    "group-ib.com", "ptsecurity.com", "cybereason.com", "sentinelone.com", "rapid7.com",
    "tenable.com", "qualys.com", "carbonblack.com", "cylance.com", "clearskysec.com",
    "intezer.com", "reversinglabs.com", "threatconnect.com", "anomali.com", "domaintools.com",
    "riskiq.com", "shadowserver.org", "abuse.ch", "alienvault.com", "otx.alienvault.com",
    "trellix.com", "netscout.com", "arbornetworks.com", "zscaler.com", "cyfirma.com",
    "securityintelligence.com", "crysys.hu", "securiteam.com", "teamviewer.com",
    # news outlets / tech blogs
    "krebsonsecurity.com", "bleepingcomputer.com", "darkreading.com", "threatpost.com",
    "zdnet.com", "wired.com", "arstechnica.com", "theregister.com", "theregister.co.uk",
    "cnet.com", "forbes.com", "reuters.com", "bbc.co.uk", "nytimes.com", "wsj.com",
    "medium.com", "wordpress.com", "wordpress.org", "blogspot.com", "blogger.com",
    "substack.com", "tumblr.com", "hackernews.com", "news.ycombinator.com",
    # developer infrastructure / social networks
    "github.com", "githubusercontent.com", "gitlab.com", "bitbucket.org", "sourceforge.net",
    "stackoverflow.com", "stackexchange.com", "reddit.com", "twitter.com", "x.com",
    "facebook.com", "linkedin.com", "youtube.com", "youtu.be", "instagram.com",
    "telegram.org", "slideshare.net", "scribd.com", "archive.org", "web.archive.org",
    # standards bodies / government / academia
    "mitre.org", "attack.mitre.org", "cve.mitre.org", "nist.gov", "nvd.nist.gov",
    "cisa.gov", "us-cert.gov", "whitehouse.gov", "fbi.gov", "justice.gov", "state.gov",
    "europa.eu", "enisa.europa.eu", "cert.org", "first.org", "sans.org", "isc.sans.edu",
    "ieee.org", "acm.org", "arxiv.org", "springer.com", "sciencedirect.com", "wikipedia.org",
    # big tech / CDNs / widely used services
    "microsoft.com", "windows.com", "office.com", "live.com", "msdn.microsoft.com",
    "technet.microsoft.com", "google.com", "googleapis.com", "gstatic.com", "goo.gl",
    "apple.com", "amazon.com", "amazonaws.com", "cloudflare.com", "akamai.com",
    "akamaihd.net", "adobe.com", "mozilla.org", "oracle.com", "ibm.com", "cisco.com",
    "vmware.com", "citrix.com", "sap.com", "salesforce.com", "dropbox.com", "box.com",
    "w3.org", "iana.org", "ietf.org", "rfc-editor.org", "gnu.org", "python.org",
    "docker.com", "kubernetes.io", "letsencrypt.org", "digicert.com", "verisign.com",
    # consulting / audit firms
    "pwc.com", "deloitte.com", "kpmg.com", "ey.com", "accenture.com", "mckinsey.com",
    "gartner.com", "forrester.com",
}

# TLDs that are never worth an API lookup
BENIGN_SUFFIXES = (".gov", ".mil", ".edu")

_IPV4 = re.compile(r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$")


def registered_domain(host: str) -> str:
    """Take the last two labels. Good enough for the list above; no PSL library needed."""
    parts = host.lower().strip(".").split(".")
    return ".".join(parts[-2:]) if len(parts) >= 2 else host.lower()


def host_of(value: str, typ: str) -> str:
    if typ == "URL":
        try:
            return (urlparse(value).hostname or "").lower()
        except ValueError:
            return ""
    return value.lower().strip(".")


def is_benign(value: str, typ: str) -> bool:
    """True when this IOC is NOT worth spending API quota on."""
    if typ == "IP":
        try:
            ip = ipaddress.ip_address(value)
        except ValueError:
            return True
        return not ip.is_global      # private / loopback / reserved / multicast / doc range
    if typ in ("Domain", "URL"):
        host = host_of(value, typ)
        if not host or _IPV4.match(host):
            return not host
        if host.endswith(BENIGN_SUFFIXES):
            return True
        rd = registered_domain(host)
        return rd in BENIGN_DOMAINS or host in BENIGN_DOMAINS
    return False


def was_defanged(value: str, text: str, typ: str) -> bool:
    """Did this IOC ever appear defanged in the source text?
    e.g. `evil[.]com`, `hxxp://...`, `1.2.3[.]4` - very strong evidence it is an IOC."""
    host = host_of(value, typ) or value
    if not host or "." not in host:
        return False
    for pat in ("[.]", "(.)", "{.}", "[dot]"):
        if host.replace(".", pat) in text:
            return True
    if typ == "URL" and ("hxxp" in text.lower()):
        tail = value.split("://", 1)[-1][:40]
        if tail and tail in text.replace("[.]", "."):
            return True
    return False


def seed_priority(ioc: dict, text: str) -> int:
    """0 = skip entirely; higher values are hunted first."""
    typ, val = ioc["type"], ioc["value"]
    if typ not in ("Domain", "IP", "URL", "Hash"):
        return 0
    if is_benign(val, typ):
        return 0
    if typ == "Hash":
        return 4          # highest: only hashes yield CMD/API nodes from the VT sandbox
    if was_defanged(val, text, typ):
        return 3
    if typ == "IP":
        return 2
    return 1              # any other domain/URL - usually just a citation


def select_seeds(iocs: list[dict], text: str, max_seeds: int = 12) -> list[dict]:
    """Filter, prioritise and truncate. Original order is preserved within a priority tier."""
    scored = []
    for i, ioc in enumerate(iocs):
        p = seed_priority(ioc, text)
        if p > 0:
            scored.append((-p, i, ioc))
    scored.sort()
    return [x[2] for x in scored[:max_seeds]]
