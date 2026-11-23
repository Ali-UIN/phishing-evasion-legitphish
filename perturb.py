"""Penyamaran (evasion) URL phishing untuk menguji ketahanan detektor.

Setiap fungsi menerima (url, rng) dan mengembalikan URL baru. Label tetap phishing:
yang berubah hanya tampilan URL, tujuan akhirnya tetap situs penipu.
Dipakai untuk riset defensif (uji ketahanan model), mengisi keterbatasan
Potpelwar dkk. (2025) yang belum memuat URL yang sengaja disamarkan.
"""
from __future__ import annotations

import random
import re
import string
from urllib.parse import urlparse, urlunparse

from features import SECOND_LEVEL

BRAND_HOSTS = [
    "www.paypal.com", "accounts.google.com", "login.microsoftonline.com",
    "www.facebook.com", "appleid.apple.com", "secure.bankmandiri.co.id",
    "ib.bri.co.id", "www.tokopedia.com",
]
SHORTENERS = ["bit.ly", "tinyurl.com", "s.id", "cutt.ly", "t.ly"]
HOMOGLYPHS = {"o": "0", "l": "1", "i": "1", "e": "3", "a": "4", "s": "5"}
_IPV4 = re.compile(r"^\d{1,3}(\.\d{1,3}){3}$")


def _split(url: str):
    raw = str(url).strip()
    if not re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", raw):
        raw = "http://" + raw
    return urlparse(raw)


def _safe(fn):
    """URL yang gagal di-parse dikembalikan apa adanya. URL asli tanpa skema
    dijaga tetap tanpa skema (kecuali force_https/shortener), supaya perubahan
    fitur benar-benar berasal dari penyamaran, bukan dari tambahan "http://"."""
    def wrapper(url, rng):
        try:
            out = fn(url, rng)
        except ValueError:
            return url
        if "://" not in str(url) and out.startswith("http://"):
            out = out[len("http://"):]
        return out
    wrapper.__name__ = fn.__name__
    return wrapper


def _host_port(p):
    host = p.hostname or ""
    port = f":{p.port}" if p.port else ""
    return host, port


def _edit_main_label(url, fn):
    p = _split(url)
    host, port = _host_port(p)
    if not host or _IPV4.match(host):
        return url  # host berupa IP tidak diubah
    labels = host.split(".")
    n_reg = 3 if ".".join(labels[-2:]) in SECOND_LEVEL else 2
    i = max(0, len(labels) - n_reg)
    labels[i] = fn(labels[i]) or labels[i]
    return urlunparse(p._replace(netloc=".".join(labels) + port))


@_safe
def force_https(url, rng):
    return urlunparse(_split(url)._replace(scheme="https"))


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
def brand_subdomain(url, rng):
    p = _split(url)
    host, port = _host_port(p)
    return urlunparse(p._replace(netloc=f"{rng.choice(BRAND_HOSTS)}.{host}{port}"))


@_safe
def at_trick(url, rng):
    p = _split(url)
    host, port = _host_port(p)
    return urlunparse(p._replace(netloc=f"{rng.choice(BRAND_HOSTS)}@{host}{port}"))


@_safe
def strip_query(url, rng):
    return urlunparse(_split(url)._replace(query="", fragment="", params=""))


@_safe
def root_landing(url, rng):
    """Halaman phishing ditaruh langsung di root domain (https://domain-palsu.com/)."""
    return urlunparse(_split(url)._replace(path="/", query="", fragment="", params=""))


@_safe
def shortener(url, rng):
    code = "".join(rng.choices(string.ascii_letters + string.digits, k=7))
    return f"https://{rng.choice(SHORTENERS)}/{code}"


def make_mimic_path(legit_paths):
    """Ganti path+query dengan path dari URL legit (meniru struktur URL asli)."""
    paths = [x for x in legit_paths if x and x != "/"] or ["/"]

    @_safe
    def mimic_legit_path(url, rng):
        p = _split(url)
        return urlunparse(p._replace(path=rng.choice(paths), query="", fragment="", params=""))
    return mimic_legit_path


def make_attacks(legit_paths):
    """Kembalikan dict nama_serangan -> fungsi(url, rng)."""
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


def apply_attack(urls, fn, seed=0):
    rng = random.Random(seed)
    return [fn(u, rng) for u in urls]


# ---------------------------------------------------------------------------
# Penyerang adaptif (black-box, berbasis kueri)
# Penyerang mencoba setiap rantai penyamaran (1 teknik, atau 2 teknik berurutan),
# menanyakan skor ke model, lalu memilih kandidat dengan probabilitas phishing
# terendah. Kandidat dibuat sekali (tidak bergantung model) supaya fiturnya cukup
# dihitung satu kali dan bisa dipakai ulang untuk semua model.
# ---------------------------------------------------------------------------
def candidate_chains(attacks):
    """Semua rantai: tiap teknik tunggal + pasangan berurutan dua teknik berbeda
    (shortener hanya tunggal karena menghapus URL asli; combo tidak dipakai)."""
    names = [a for a in attacks if a != "combo"]
    stack = [a for a in names if a != "shortener"]
    return [(a,) for a in names] + [(a, b) for a in stack for b in stack if a != b]


def generate_candidates(urls, attacks, chains, seed=0):
    """Kembalikan (url_kandidat, indeks_url_asal, indeks_rantai)."""
    import numpy as np
    rng = random.Random(seed)
    out, owner, chain_id = [], [], []
    for i, u in enumerate(urls):
        for j, ch in enumerate(chains):
            v = u
            for name in ch:
                v = attacks[name](v, rng)
            out.append(v)
            owner.append(i)
            chain_id.append(j)
    return out, np.asarray(owner), np.asarray(chain_id)
