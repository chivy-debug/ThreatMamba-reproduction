"""IOCHunter: hunting-method scoring following Eq. 1 of the paper.

Eq. 1 verbatim:

    s_{m,o} = C_m / (T_m + 1e-6)  +  alpha * sqrt( (ln( sum_{n in H_{M,o}} T_n ) + 1e-6) / T_m )

  C_m  = Coverage, the total number of new nodes obtained from method m
  T_m  = Tried Time, how many times method m has been called
  H_{M,o} = the set of methods APPLICABLE to IOC o. The sum in the numerator runs over this
            set only, NOT over every method.
  alpha = exploration weight

Note the bare `T_m` in the denominator (not `T_m + 1`): while a method has never been
called the exploration term is infinite, so Eq. 1 FORCES every method to be tried at least
once before exploitation begins. This is the anti "early-stage bias" mechanism the paper
describes just before Eq. 1.

The hunting-method table follows Table I of the paper.
"""
import math

INF = float("inf")

# method -> (applicable IOC types, description shown to the LLM in the Fig. 2 prompt, source)
METHODS: dict[str, tuple[tuple[str, ...], str, str]] = {
    "rapiddns_history": (
        ("Domain", "IP"),
        "Lookup historical DNS records of an IP or Domain; returns sibling domains "
        "hosted on the same IP, and first/last seen timestamps.",
        "RapidDNS"),
    "otx_general": (
        ("IP", "Domain", "URL", "Hash"),
        "Query AlienVault OTX indicator general + passive DNS; returns linked indicators, "
        "geolocation of the hosting infrastructure, and observation timestamps.",
        "AlienVault OTX"),
    "vt_object": (
        ("URL", "Hash", "Domain", "IP"),
        "Fetch the VirusTotal object details of a URL/file/domain/IP; returns contacted "
        "domains and IPs, country, and submission timestamps.",
        "VirusTotal"),
    "vt_behavior": (
        ("Hash",),
        "Fetch the VirusTotal sandbox behaviour report of a file hash; returns executed "
        "command lines and Windows API calls observed at runtime.",
        "VirusTotal"),
}


class UCTSelector:
    def __init__(self, alpha: float = 1.414):
        self.alpha = alpha
        self.tried = {m: 0 for m in METHODS}
        self.coverage = {m: 0 for m in METHODS}

    def applicable(self, ioc_type: str) -> list[str]:
        return [m for m, (types, _d, _s) in METHODS.items() if ioc_type in types]

    def describe(self, m: str) -> dict:
        """Describe a method in the "Hunting Methods" format used by the Fig. 2 prompt."""
        _types, desc, source = METHODS[m]
        return {"Method": m, "Source": source, "Description": desc,
                "Coverage": self.coverage[m], "Tried Time": self.tried[m]}

    def score(self, m: str, ioc_type: str | None = None) -> float:
        """Eq. 1. A method that has never been called scores +inf (forced exploration)."""
        t_m = self.tried[m]
        if t_m == 0:
            return INF
        peers = self.applicable(ioc_type) if ioc_type else list(METHODS)
        n_total = sum(self.tried[p] for p in peers)
        exploit = self.coverage[m] / (t_m + 1e-6)
        explore = self.alpha * math.sqrt((math.log(max(n_total, 1)) + 1e-6) / t_m)
        return exploit + explore

    def rank(self, ioc_type: str, exclude: set[str] | None = None, top_k: int = 50) -> list[str]:
        """Algorithm 1, lines 18-21: sort methods by descending s_{m,o} and take the top
        k=50. Methods already tried for this IOC are excluded (lines 5-10: avoid re-hunting)."""
        cands = [m for m in self.applicable(ioc_type) if not exclude or m not in exclude]
        return sorted(cands, key=lambda m: self.score(m, ioc_type), reverse=True)[:top_k]

    def select(self, ioc_type: str, exclude: set[str] | None = None) -> str | None:
        best = self.rank(ioc_type, exclude, top_k=1)
        return best[0] if best else None

    def update(self, m: str, gain: int):
        """gain = number of new nodes obtained from this call (Coverage in Eq. 1)."""
        self.tried[m] += 1
        self.coverage[m] += max(0, int(gain))

    def state(self) -> dict:
        return {"tried": dict(self.tried), "coverage": dict(self.coverage)}

    def load_state(self, st: dict):
        for m in METHODS:
            self.tried[m] = int(st.get("tried", {}).get(m, 0))
            self.coverage[m] = int(st.get("coverage", {}).get(m, 0))
