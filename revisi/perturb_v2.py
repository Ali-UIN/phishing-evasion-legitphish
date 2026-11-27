"""URL disguising techniques for the revision (pipeline v2).

Same nine base techniques and combo as perturb.py, with one change: typosquat and
homoglyph locate the registered-domain label with the Public Suffix List.
Three extension techniques answer Reviewer 2, comment 5. They are evaluated as a
separate experiment and are NOT part of the main pool of 65 chains:
- idn_homoglyph: Latin letters replaced by Cyrillic look-alikes, label written
  in its Punycode form (xn--...), as it appears in a URL string.
- open_redirect: the phishing URL is passed as a parameter of a redirect endpoint
  on a well-known domain.
- custom_shortlink: a short link on a random short domain instead of one of the
  five fixed shortener domains.
All of this is for defensive robustness testing; the label stays phishing.
"""
from __future__ import annotations

import itertools
import os
import random
import string
import sys
from urllib.parse import quote, urlunparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from perturb import (HOMOGLYPHS, _IPV4, _host_port, _safe, _split, at_trick,  # noqa: E402
                     brand_subdomain, candidate_chains, force_https, make_mimic_path,
                     root_landing, shortener, strip_query)
from psl import split_host  # noqa: E402

CYRILLIC = {"a": "а", "c": "с", "e": "е", "o": "о", "p": "р",
            "x": "х", "y": "у", "i": "і", "s": "ѕ", "j": "ј"}
OPEN_REDIRECTS = [
    "https://www.google.com/url?q={u}",
    "https://l.facebook.com/l.php?u={u}",
    "https://www.youtube.com/redirect?q={u}",
    "https://www.linkedin.com/redir/redirect?url={u}",
    "https://vk.com/away.php?to={u}",
]
SHORT_TLDS = ["link", "to", "ly", "co", "me", "io", "gg", "page"]


def _edit_main_label(url, fn):
    p = _split(url)
    host, port = _host_port(p)
    if not host or _IPV4.match(host):
        return url  # an IP host is left unchanged
    sub, main, suffix = split_host(host)
    if not main:
        return url  # the host is a bare public suffix
    new = fn(main) or main
    return urlunparse(p._replace(netloc=".".join([*sub, new, suffix]) + port))


@_safe
def typosquat(url, rng):
    def op(label):
        if len(label) < 4:
            return label + label[-1]
        j = rng.randrange(1, len(label) - 1)
        kind = rng.choice(["double", "swap", "drop"])
        if kind == "double":
            return label[:j] + label[j] + label[j:]
        if kind == "swap":
            return label[:j - 1] + label[j] + label[j - 1] + label[j + 1:]
        return label[:j] + label[j + 1:]
    return _edit_main_label(url, op)


@_safe
def homoglyph(url, rng):
    def op(label):
        idx = [k for k, ch in enumerate(label) if ch in HOMOGLYPHS]
        rng.shuffle(idx)
        chars = list(label)
        for k in idx[:2]:
            chars[k] = HOMOGLYPHS[chars[k]]
        return "".join(chars)
    return _edit_main_label(url, op)


@_safe
def idn_homoglyph(url, rng):
    def op(label):
        if label.startswith("xn--"):
            return label
        idx = [k for k, ch in enumerate(label) if ch in CYRILLIC]
        if not idx:
            return label
        rng.shuffle(idx)
        chars = list(label)
        for k in idx[:2]:
            chars[k] = CYRILLIC[chars[k]]
        return "xn--" + "".join(chars).encode("punycode").decode("ascii")
    return _edit_main_label(url, op)


def open_redirect(url, rng):
    target = str(url).strip()
    if "://" not in target:
        target = "http://" + target
    return rng.choice(OPEN_REDIRECTS).format(u=quote(target, safe=""))


def custom_shortlink(url, rng):
    label = "".join(rng.choices(string.ascii_lowercase, k=rng.randint(3, 6)))
    code = "".join(rng.choices(string.ascii_letters + string.digits, k=rng.randint(5, 8)))
    return f"https://{label}.{rng.choice(SHORT_TLDS)}/{code}"


EXT_ATTACKS = {"idn_homoglyph": idn_homoglyph, "open_redirect": open_redirect,
               "custom_shortlink": custom_shortlink}


def make_attacks(legit_paths):
    """name -> function(url, rng), same order as perturb.make_attacks."""
    attacks = {
        "force_https": force_https,
        "root_landing": root_landing,
        "typosquat": typosquat,
        "homoglyph": homoglyph,
        "brand_subdomain": brand_subdomain,
        "at_trick": at_trick,
        "strip_query": strip_query,
        "mimic_legit_path": make_mimic_path(legit_paths),
        "shortener": shortener,
    }
    stackable = [k for k in attacks if k != "shortener"]

    def combo(url, rng):
        for name in rng.sample(stackable, k=3):
            url = attacks[name](url, rng)
        return url
    attacks["combo"] = combo
    return attacks


STACKABLE = ["force_https", "root_landing", "typosquat", "homoglyph", "brand_subdomain",
             "at_trick", "strip_query", "mimic_legit_path"]


def triple_chains(n=56, seed=2026):
    """n ordered chains of three different stackable techniques (out of 336)."""
    allc = list(itertools.permutations(STACKABLE, 3))
    return random.Random(seed).sample(allc, n)


def extension_chains():
    """The three extension techniques alone, plus idn_homoglyph paired (both orders)
    with the six stackable techniques that do not edit the domain label."""
    others = [a for a in STACKABLE if a not in ("typosquat", "homoglyph")]
    return ([(a,) for a in EXT_ATTACKS]
            + [("idn_homoglyph", b) for b in others]
            + [(b, "idn_homoglyph") for b in others])


def split_pair_families(seed):
    """Split the 28 unordered pairs of stackable techniques in half. Returns the set
    of pairs (frozensets) whose two orderings are withheld from adversarial training."""
    pairs = [frozenset(c) for c in itertools.combinations(STACKABLE, 2)]
    random.Random(seed).shuffle(pairs)
    return set(pairs[len(pairs) // 2:])


__all__ = ["make_attacks", "EXT_ATTACKS", "candidate_chains", "triple_chains",
           "extension_chains", "split_pair_families", "STACKABLE"]
