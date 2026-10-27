"""How far a web source can be believed, decided by rules (no network, no model).

A source gets a tier from its domain (T1 primary/official, T2 established reference or news, T3 community
or user-written, T4 unknown) and loses points for what the page itself shows: affiliate links, clickbait
"top 10" titles, no author, no date, thin text, ad walls. Look-alikes of well-known domains and pages that
try to instruct the assistant are rated "avoid" outright. The result is a level high / medium / low / avoid
plus the reasons, so the model (and the GUI) can say *why* a source is weak.

There is deliberately no built-in blacklist of "bad" sites: judging a site as a misinformation source is
the user's call, made through ~/.config/agent/source-trust.yaml or the GUI (allow / deny / tiers).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

TIER_LEVEL = {"T1": 3, "T2": 2, "T3": 1, "T4": 1}

_T1 = (
    "gov", "gov.uk", "gov.au", "edu", "ac.uk", "mil", "int", "bund.de", "europa.eu",
    "w3.org", "ietf.org", "iso.org", "ieee.org", "nist.gov", "who.int", "doi.org", "arxiv.org", "nature.com",
    "science.org", "ncbi.nlm.nih.gov", "python.org", "rust-lang.org", "kernel.org", "mozilla.org", "gnu.org",
    "nixos.org", "archlinux.org", "debian.org", "postgresql.org", "sqlite.org", "docs.rs", "crates.io",
    "pypi.org", "npmjs.com", "golang.org", "go.dev", "ollama.com", "learn.microsoft.com", "developer.apple.com",
    "developer.mozilla.org", "cloud.google.com", "docs.aws.amazon.com", "kubernetes.io", "llvm.org", "khronos.org",
)
_T2 = (
    "wikipedia.org", "britannica.com", "reuters.com", "apnews.com", "bbc.com", "bbc.co.uk", "nytimes.com",
    "theguardian.com", "dw.com", "tagesschau.de", "zdf.de", "spiegel.de", "zeit.de", "faz.net", "heise.de",
    "arstechnica.com", "wired.com", "github.com", "gitlab.com", "huggingface.co", "wiki.archlinux.org",
    "nixos.wiki", "phoronix.com", "open-meteo.com", "frankfurter.app", "lwn.net", "tomshardware.com", "techpowerup.com",
)
_T3 = (
    "stackoverflow.com", "stackexchange.com", "superuser.com", "askubuntu.com", "serverfault.com", "reddit.com",
    "quora.com", "medium.com", "dev.to", "substack.com", "wordpress.com", "blogspot.com", "news.ycombinator.com",
    "tumblr.com", "wixsite.com", "fandom.com", "answers.com",
)
_TERTIARY = ("wikipedia.org", "britannica.com", "fandom.com")
_PREPRINT = ("arxiv.org",)
_TWO_PART_SUFFIXES = frozenset({
    "co.uk", "ac.uk", "gov.uk", "org.uk", "com.au", "gov.au", "co.jp", "com.br", "co.nz", "co.at", "gv.at", "or.at",
})

_AFFILIATE = re.compile(
    r"(?i)(amzn\.to|[?&]tag=[\w-]+-2[01]\b|[?&]tag=\w+|awin1\.com|shareasale\.com|clickbank|impact\.com|"
    r"as an amazon associate|we may earn (a|an affiliate) commission|affiliate (link|commission|disclosure)|"
    r"partnerize|tradedoubler|jetzt kaufen bei amazon|\*affiliate)"
)
_CLICKBAIT = re.compile(
    r"(?i)(^\s*(the\s+)?(top|best)\s+\d+\b|\b\d+\s+best\b|\bbest\b.{0,40}\b20\d\d\b|you won'?t believe|"
    r"this one (simple )?trick|doctors (hate|don'?t want)|what happens next|shocking|jaw[- ]dropping|"
    r"\bcheap(est)?\b.{0,30}\bdeals?\b|bestenliste|im test:|die besten \d+)"
)
_BYLINE_HTML = re.compile(
    r'(?is)(rel=["\']author["\']|<meta[^>]+name=["\']author["\']|itemprop=["\']author["\']|'
    r'"author"\s*:\s*[{\["]|class=["\'][^"\']*(byline|author)[^"\']*["\'])'
)
_BYLINE_TEXT = re.compile(r"(?i)^\W*(by|von|autor|written by)\s+[A-ZÄÖÜ][\w.'-]+(\s+[A-ZÄÖÜ][\w.'-]+)?", re.M)
_ADWORDS = re.compile(r"(?i)\b(advertisement|sponsored|anzeige|werbung|promoted)\b")


@dataclass
class Trust:
    tier: str = "T4"
    level: str = "low"
    reasons: list[str] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)
    user_rule: str | None = None

    def label(self) -> str:
        why = ", ".join(self.reasons)
        return f"{self.tier} {self.level}" + (f": {why}" if why else "")

    def as_dict(self) -> dict:
        return {"tier": self.tier, "level": self.level, "reasons": list(self.reasons),
                "flags": list(self.flags), "user_rule": self.user_rule}


@dataclass
class Rules:
    allow: frozenset[str] = frozenset()
    deny: frozenset[str] = frozenset()
    tiers: dict[str, str] = field(default_factory=dict)


RULES_PATH = Path.home() / ".config" / "agent" / "source-trust.yaml"
_rules_cache: tuple[float, Rules] | None = None


def load_rules(path: Path | None = None) -> Rules:
    """The user's allow / deny / tier overrides (empty when the file is missing or unreadable)."""
    global _rules_cache
    path = path or RULES_PATH
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return Rules()
    if path == RULES_PATH and _rules_cache and _rules_cache[0] == mtime:
        return _rules_cache[1]
    try:
        data = yaml.safe_load(path.read_text()) or {}
        rules = Rules(
            allow=frozenset(registrable(str(d)) for d in data.get("allow") or []),
            deny=frozenset(registrable(str(d)) for d in data.get("deny") or []),
            tiers={registrable(str(k)): str(v).upper() for k, v in (data.get("tiers") or {}).items()
                   if str(v).upper() in TIER_LEVEL},
        )
    except (OSError, yaml.YAMLError, AttributeError):
        return Rules()
    if path == RULES_PATH:
        _rules_cache = (mtime, rules)
    return rules


def registrable(domain: str) -> str:
    """`blog.example.co.uk` -> `example.co.uk` (small suffix table instead of the full public-suffix list)."""
    host = domain.lower().strip().removeprefix("www.").split("/")[0]
    labels = host.split(".")
    if len(labels) >= 3 and ".".join(labels[-2:]) in _TWO_PART_SUFFIXES:
        return ".".join(labels[-3:])
    return ".".join(labels[-2:]) if len(labels) >= 2 else host


def _matches(host: str, entries: tuple[str, ...]) -> bool:
    return any(host == e or host.endswith("." + e) for e in entries)


def _distance(a: str, b: str, limit: int = 2) -> int:
    """Edit distance, giving up above `limit`."""
    if abs(len(a) - len(b)) > limit:
        return limit + 1
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def lookalike_of(host: str) -> str:
    """The well-known domain `host` imitates (one or two characters off), or ""."""
    reg = registrable(host)
    label = reg.split(".")[0].replace("-", "")
    if len(label) < 6:
        return ""
    for known in _T1 + _T2:
        k = registrable(known)
        klabel = k.split(".")[0]
        if k != reg and len(klabel) >= 6 and _distance(label, klabel, 1 if len(klabel) < 10 else 2) <= (1 if len(klabel) < 10 else 2):
            return k
    return ""


def domain_tier(host: str, rules: Rules | None = None) -> tuple[str, str]:
    """(tier, reason) from the domain alone."""
    reg = registrable(host)
    if rules and reg in rules.tiers:
        return rules.tiers[reg], "tier set by you"
    if _matches(host, _T1) or host.rsplit(".", 1)[-1] in ("gov", "edu", "mil", "int"):
        return "T1", "official or primary source"
    if _matches(host, _T2):
        return "T2", "established reference or news"
    if _matches(host, _T3):
        return "T3", "community or user-written"
    return "T4", "unknown site"


def page_signals(html: str) -> dict:
    """What the raw page shows about itself; small enough to cache next to the cleaned text."""
    return {"affiliate": bool(_AFFILIATE.search(html)), "byline": bool(_BYLINE_HTML.search(html)),
            "ads": len(_ADWORDS.findall(_COMMENTS_RE.sub(" ", html)))}


_COMMENTS_RE = re.compile(r"(?s)<!--.*?-->")


def assess(url: str, title: str = "", html: str = "", text: str = "", published: str = "",
           rules: Rules | None = None, injection: bool = False, signals: dict | None = None) -> Trust:
    """Rate one source. `signals` (from `page_signals`, cached) or the raw `html` give the best evidence;
    `text` alone still works (byline, ad words, length)."""
    sig = signals if signals is not None else (page_signals(html) if html else {})
    host = re.sub(r"^https?://", "", url.lower()).split("/")[0].split(":")[0].removeprefix("www.")
    reg = registrable(host)
    rules = rules if rules is not None else load_rules()
    tier, why = domain_tier(host, rules)
    reasons: list[str] = [why] if tier != "T4" else []
    flags: list[str] = []

    if rules.deny and reg in rules.deny:
        return Trust(tier, "avoid", ["blocked by you"], ["user_denied"], "deny")
    if injection:
        return Trust(tier, "avoid", ["page tried to give the assistant instructions"], ["suspicious_instructions"])
    imitated = lookalike_of(host)
    if imitated:
        return Trust("T4", "avoid", [f"looks like {imitated} but is not"], ["typosquat"])

    score = TIER_LEVEL[tier]
    if _matches(host, _TERTIARY):
        flags.append("tertiary")
    if _matches(host, _PREPRINT):
        flags.append("preprint")
        reasons.append("preprint, not peer-reviewed")
    bad = 0
    if sig.get("affiliate"):
        flags.append("affiliate"); reasons.append("affiliate links"); bad += 1
    if title and _CLICKBAIT.search(title):
        flags.append("clickbait"); reasons.append("clickbait or listicle title"); bad += 1
    if sig.get("ads", 0) + len(_ADWORDS.findall(text)) >= 4:
        flags.append("ad_heavy"); reasons.append("ad-heavy page"); bad += 1
    if text and len(text) < 400:
        flags.append("thin"); reasons.append("very little text"); bad += 1
    if url.startswith("http://"):
        flags.append("http_only")
    has_byline = bool(sig.get("byline")) or bool(_BYLINE_TEXT.search((text or "")[:600]))
    soft = 0
    if not has_byline:
        flags.append("no_author"); soft += 1
    if not published:
        flags.append("undated"); soft += 1
    if tier in ("T3", "T4"):
        if "no_author" in flags:
            reasons.append("no author")
        if "undated" in flags:
            reasons.append("undated")
        if tier == "T4" and not bad and not soft:
            score += 1                                # unknown site that names its author, dates its page, sells nothing
    score -= bad
    if rules.allow and reg in rules.allow:
        score = max(score, 2)
        reasons.append("allowed by you")
    level = "high" if score >= 3 else "medium" if score == 2 else "low" if score >= 0 else "avoid"
    return Trust(tier, level, reasons, flags, "allow" if reg in rules.allow else None)
