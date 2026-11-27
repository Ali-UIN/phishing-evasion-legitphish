"""Feature extraction for the revision (pipeline v2).

Changes against features.py (which stays untouched for the original notebook):
- canonical_key(): RFC 3986 syntax-based normalisation, used to remove
  functionally identical URLs before the train/test split (Reviewer 1, comment 1).
- The corrected feature set takes the public suffix and the registered domain
  from the Mozilla Public Suffix List instead of a short manual list
  (Reviewer 1, comment 2). Feature names and meanings are unchanged.
- Three host-based features that an attacker cannot change by editing the scheme,
  path, or query (Reviewer 2, comment 3): domain_entropy, brand_distance,
  brand_in_subdomain.

One row = 16 replicated features + 18 corrected features + 3 host features.
"""
from __future__ import annotations

import os
import re
import string
import sys
from collections import Counter
from functools import lru_cache
from urllib.parse import urlparse

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from features import (FEATURES, FEATURES_FIXED, POPULAR_TLDS, SUSPICIOUS_EXT_FIXED,  # noqa: E402
                      _ALNUM_SPLIT, _IPV4, _entropy, _fixed_parse, _host)
from features import extract as extract_repl  # noqa: E402
from psl import registered_domain_of_host, split_host  # noqa: E402

HOST_FEATURES = ["domain_entropy", "brand_distance", "brand_in_subdomain"]
ALL_COLS = ["r_" + f for f in FEATURES] + FEATURES_FIXED + HOST_FEATURES
COLS_REPL = list(range(len(FEATURES)))
COLS_FIXED = list(range(len(FEATURES), len(FEATURES) + len(FEATURES_FIXED)))
COLS_HOST = list(range(len(FEATURES) + len(FEATURES_FIXED), len(ALL_COLS)))

# Popular brand names for the two brand features (global services plus Indonesian
# banks and marketplaces). Independent of the attack code, although it covers the
# eight brand hostnames that perturb.py uses.
BRANDS = [
    "paypal", "google", "microsoft", "microsoftonline", "apple", "appleid", "facebook",
    "amazon", "netflix", "instagram", "whatsapp", "linkedin", "twitter", "yahoo",
    "outlook", "office365", "dropbox", "adobe", "ebay", "chase", "wellsfargo",
    "bankofamerica", "citibank", "hsbc", "fedex", "steam", "binance", "coinbase",
    "metamask", "telegram", "tiktok", "shopee", "tokopedia", "bukalapak", "gojek",
    "bankmandiri", "mandiri", "klikbca", "bri", "bni", "bca", "dana", "dhl", "usps",
]
_BRANDS_DIST = [b for b in BRANDS if len(b) >= 5]
_TOKEN = re.compile(r"[^a-z0-9]+")

# ---------------------------------------------------------------------------
# Canonical form for duplicate removal
# ---------------------------------------------------------------------------
_SCHEME = re.compile(r"^([a-zA-Z][a-zA-Z0-9+.-]*)://")
_PCT = re.compile(r"%([0-9a-fA-F]{2})")
_UNRESERVED = frozenset(string.ascii_letters + string.digits + "-._~")
_DEFAULT_PORT = {"http": 80, "https": 443, "ftp": 21}


def _norm_pct(s: str) -> str:
    """Decode percent-escapes of unreserved characters, upper-case the others."""
    def rep(m):
        ch = chr(int(m.group(1), 16))
        return ch if ch in _UNRESERVED else "%" + m.group(1).upper()
    return _PCT.sub(rep, s)


def _remove_dot_segments(path: str) -> str:
    if not path:
        return "/"
    segs = path.split("/")
    out = []
    for i, s in enumerate(segs):
        last = i == len(segs) - 1
        if s == ".":
            if last:
                out.append("")
        elif s == "..":
            if len(out) > 1:
                out.pop()
            if last:
                out.append("")
        else:
            out.append(s)
    res = "/".join(out)
    return res if res.startswith("/") else "/" + res


def canonical_key(url: str, aggressive: bool = False) -> str:
    """RFC 3986 section 6.2.2 and 6.2.3 normalisation: lower-case scheme and host,
    drop the default port, normalise percent-encoding, remove dot segments, and
    turn an empty path into "/". A URL without a scheme keeps that property.

    aggressive=True also treats a missing scheme as http, drops the fragment and
    strips a trailing slash. It is only used to report how many more URLs would
    collide under a looser rule; it is not used for cleaning."""
    raw = str(url).strip()
    m = _SCHEME.match(raw)
    try:
        p = urlparse(raw if m else "http://" + raw)
        host = (p.hostname or "").lower()
        port = p.port
    except ValueError:
        return raw
    scheme = p.scheme.lower() if m else ""
    netloc = f"[{host}]" if ":" in host else host
    if "@" in p.netloc:
        netloc = p.netloc.rsplit("@", 1)[0] + "@" + netloc
    if port is not None and port != _DEFAULT_PORT.get(scheme or "http"):
        netloc += f":{port}"
    path = _remove_dot_segments(_norm_pct(p.path))
    query = _norm_pct(p.query)
    frag = _norm_pct(p.fragment)
    if aggressive:
        scheme = scheme or "http"
        frag = ""
        path = path.rstrip("/") or "/"
    key = (scheme + "://" if scheme else "//") + netloc + path
    if p.params:
        key += ";" + p.params
    if query:
        key += "?" + query
    if frag:
        key += "#" + frag
    return key


def registered_domain_v2(url: str) -> str:
    """Registered domain by the Public Suffix List (login.bank.co.id -> bank.co.id)."""
    host = _host(str(url).strip())
    if not host or _IPV4.match(host):
        return host
    return registered_domain_of_host(host)


# ---------------------------------------------------------------------------
# Host-based features
# ---------------------------------------------------------------------------
def _lev(a: str, b: str) -> int:
    if a == b:
        return 0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


_BRAND_INFO = [(b, len(b), frozenset(b), Counter(b)) for b in _BRANDS_DIST]
_DIST_CAP = 0.5


@lru_cache(maxsize=2_000_000)
def brand_distance(label: str) -> float:
    """Smallest normalised Levenshtein distance between a brand name and the
    registered-domain label or one of its hyphen-separated parts (0 = a brand
    name itself, small = look-alike). Distances above 0.5 carry no look-alike
    signal and are reported as 1. Cheap lower bounds (length and character
    counts) skip most of the exact computations."""
    if not label:
        return 1.0
    best = _DIST_CAP + 1e-9
    for part in {label, *label.split("-")}:
        lp = len(part)
        if lp < 3:
            continue
        sp, cp = frozenset(part), None
        for b, lb, sb, cb in _BRAND_INFO:
            n = max(lp, lb)
            if max(abs(lp - lb), len(sp - sb), len(sb - sp)) / n >= best:
                continue
            if cp is None:
                cp = Counter(part)
            if max(sum((cp - cb).values()), sum((cb - cp).values())) / n >= best:
                continue
            d = _lev(part, b) / n
            if d < best:
                best = d
    return best if best <= _DIST_CAP else 1.0


@lru_cache(maxsize=1_000_000)
def brand_in_subdomain(sub: str) -> int:
    """1 if a brand name appears in the labels in front of the registered domain."""
    for tok in _TOKEN.split(sub):
        if not tok:
            continue
        for b in BRANDS:
            if tok == b or (len(b) >= 6 and b in tok):
                return 1
    return 0


def extract_v2(url: str) -> list:
    """18 corrected features (Public Suffix List parsing) followed by 3 host features."""
    raw = str(url).strip()
    p, host, port, no_scheme = _fixed_parse(raw)
    is_ip = bool(_IPV4.match(host))
    if is_ip or not host:
        sub, main, suffix = [], "", ""
    else:
        sub, main, suffix = split_host(host)
    digits = sum(ch.isdigit() for ch in raw)
    path = p.path or ""
    return [
        len(raw),                                                       # url_length
        int(is_ip),                                                     # has_ip_address
        raw.count("."),                                                 # dot_count
        int(p.scheme.lower() == "https" and not no_scheme),             # https_flag
        _entropy(raw),                                                  # url_entropy
        len([t for t in _ALNUM_SPLIT.split(raw) if t]),                 # token_count
        len(sub),                                                       # subdomain_count
        len([q for q in p.query.split("&") if q]),                      # query_param_count
        len(suffix),                                                    # tld_length
        len(path),                                                      # path_length
        int("-" in host),                                               # has_hyphen_in_domain
        digits,                                                         # number_of_digits
        int(suffix.split(".")[-1] in POPULAR_TLDS) if suffix else 0,    # tld_popularity
        int(path.lower().endswith(SUSPICIOUS_EXT_FIXED)),               # suspicious_file_extension
        len(main),                                                      # domain_name_length
        (digits / len(raw) * 100) if raw else 0.0,                      # percentage_numeric_chars
        int(port is not None),                                          # has_port
        int(no_scheme),                                                 # no_scheme
        _entropy(main),                                                 # domain_entropy
        brand_distance(main),                                           # brand_distance
        brand_in_subdomain(".".join(sub)),                              # brand_in_subdomain
    ]


def extract_rows(urls) -> np.ndarray:
    """All 37 columns (see ALL_COLS) for a list of URLs, as float32."""
    out = np.empty((len(urls), len(ALL_COLS)), dtype=np.float32)
    for i, u in enumerate(urls):
        d = extract_repl(u)
        out[i] = [d[f] for f in FEATURES] + extract_v2(u)
    return out
