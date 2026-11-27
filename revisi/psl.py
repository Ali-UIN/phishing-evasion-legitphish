"""Registered-domain parsing with the Mozilla Public Suffix List (PSL).

Uses the offline snapshot psl_snapshot.txt (see buat_psl_snapshot.py for its origin)
and the standard PSL matching algorithm: the longest matching rule wins, exception
rules ("!") beat wildcard rules ("*."), and an unlisted TLD counts as a one-label
suffix. Only the ICANN section is used by default, which is also the default of
tldextract; private suffixes such as blogspot.com are treated as ordinary domains.
"""
from __future__ import annotations

import os
from functools import lru_cache

SNAPSHOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "psl_snapshot.txt")


def _ascii(rule: str) -> str:
    """Punycode form of an IDN rule, keeping a leading '*.' or '!'."""
    head = ""
    if rule.startswith("!"):
        head, rule = "!", rule[1:]
    elif rule.startswith("*."):
        head, rule = "*.", rule[2:]
    return head + rule.encode("idna").decode("ascii")


def load_rules(path: str = SNAPSHOT, private: bool = False):
    exact, wild, exc = set(), set(), set()
    section = "icann"
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("// ===BEGIN PRIVATE"):
                section = "private"
            if not line or line.startswith("//"):
                continue
            if section == "private" and not private:
                continue
            rule = line.split()[0].lower()
            forms = {rule}
            if not rule.isascii():
                try:
                    forms.add(_ascii(rule))
                except UnicodeError:
                    pass
            for r in forms:
                if r.startswith("!"):
                    exc.add(r[1:])
                elif r.startswith("*."):
                    wild.add(r[2:])
                else:
                    exact.add(r)
    return exact, wild, exc


_EXACT, _WILD, _EXC = load_rules()


@lru_cache(maxsize=500_000)
def suffix_labels(host: str) -> int:
    """Number of labels of `host` that belong to its public suffix."""
    labels = host.split(".")
    n = len(labels)
    for i in range(n):
        cand = ".".join(labels[i:])
        if cand in _EXC:
            return n - i - 1
        if cand in _EXACT:
            return n - i
        if i + 1 < n and ".".join(labels[i + 1:]) in _WILD:
            return n - i
    return 1


def split_host(host: str):
    """Return (subdomain_labels, registered_label, public_suffix) for a lowercase hostname.
    If the host is itself a public suffix, registered_label is ''."""
    labels = [l for l in host.split(".") if l]
    if not labels:
        return [], "", ""
    k = suffix_labels(".".join(labels))
    suffix = ".".join(labels[-k:])
    if len(labels) <= k:
        return [], "", suffix
    return labels[:-k - 1], labels[-k - 1], suffix


def registered_domain_of_host(host: str) -> str:
    sub, main, suffix = split_host(host)
    return f"{main}.{suffix}" if main else suffix
