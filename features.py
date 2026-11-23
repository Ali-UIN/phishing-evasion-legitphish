"""Ekstraksi 16 fitur leksikal URL, mereplikasi definisi dataset LegitPhish
(Potpelwar dkk., 2025, Data in Brief 63, 111972).

Paper hanya memberi deskripsi singkat tiap fitur (Tabel 1), jadi rumus persisnya
direkonstruksi dari data: dicocokkan dengan kolom bawaan pada sampel 30.000 URL
dan hasilnya sama 100% untuk ke-16 fitur (domain_name_length 99,99%).

Catatan perilaku asli yang sengaja dipertahankan supaya sama dengan dataset:
- netloc diambil dari urlparse(url) apa adanya (tanpa menambah skema), jadi URL
  tanpa "http://" punya netloc kosong dan seluruh URL dianggap path.
- netloc dipecah pakai titik secara polos: port ikut terbawa ke "TLD",
  ".co.uk" dianggap TLD "uk" dengan domain "co", dan subdomain_count bisa -1.
- query_param_count = jumlah bagian query.split("&"), jadi URL tanpa query = 1.

Fitur dihitung ulang dari URL mentah (bukan memakai kolom bawaan) supaya URL hasil
penyamaran (perturb.py) diproses dengan rumus yang persis sama.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from urllib.parse import urlparse

import pandas as pd

FEATURES = [
    "url_length",
    "has_ip_address",
    "dot_count",
    "https_flag",
    "url_entropy",
    "token_count",
    "subdomain_count",
    "query_param_count",
    "tld_length",
    "path_length",
    "has_hyphen_in_domain",
    "number_of_digits",
    "tld_popularity",
    "suspicious_file_extension",
    "domain_name_length",
    "percentage_numeric_chars",
]

# Dipakai perturb.py untuk memilih label domain inti, dan registered_domain()
# untuk split per domain. Tidak dipakai dalam rumus fitur.
SECOND_LEVEL = {
    "co.id", "ac.id", "go.id", "or.id", "web.id", "my.id", "sch.id", "net.id",
    "co.uk", "org.uk", "ac.uk", "gov.uk", "com.au", "net.au", "org.au",
    "co.jp", "ne.jp", "or.jp", "com.br", "com.cn", "com.my", "com.sg",
    "co.in", "co.kr", "com.tr", "com.mx", "co.za",
}

POPULAR_TLDS = {"com", "org", "net", "edu", "gov"}           # direkonstruksi dari data
SUSPICIOUS_ENDINGS = ("exe", "zip", "bin", "apk", "cmd", "js")  # path.endswith, tanpa titik

_IPV4 = re.compile(r"^\d{1,3}(\.\d{1,3}){3}$")
_TOKEN_SPLIT = re.compile(r"[/.?=&]")
_EMPTY = urlparse("")


def _raw_parse(url: str):
    """urlparse apa adanya, seperti di dataset asli."""
    try:
        return urlparse(url)
    except ValueError:  # URL rusak (misal IPv6 tidak valid)
        return _EMPTY


def _host(url: str) -> str:
    """Hostname setelah menambah skema bila perlu (untuk deteksi IP & domain)."""
    to_parse = url if re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", url) else "http://" + url
    try:
        return (urlparse(to_parse).hostname or "").lower()
    except ValueError:
        return ""


def registered_domain(url: str) -> str:
    """Domain terdaftar sederhana (contoh: login.bank.co.id -> bank.co.id).
    Hanya untuk split per domain, bukan fitur model."""
    host = _host(str(url).strip())
    if not host or _IPV4.match(host):
        return host
    labels = host.split(".")
    n = 3 if ".".join(labels[-2:]) in SECOND_LEVEL else 2
    return ".".join(labels[-n:])


def _entropy(s: str) -> float:
    if not s:
        return 0.0
    counts = Counter(s)
    n = len(s)
    return -sum(c / n * math.log2(c / n) for c in counts.values())


def extract(url: str) -> dict:
    raw = str(url).strip()
    p = _raw_parse(raw)
    netloc = p.netloc
    parts = netloc.split(".")
    digits = sum(ch.isdigit() for ch in raw)
    return {
        "url_length": len(raw),
        "has_ip_address": int(bool(_IPV4.match(_host(raw)))),
        "dot_count": raw.count("."),
        "https_flag": int(p.scheme.lower() == "https"),
        "url_entropy": _entropy(raw),
        "token_count": len(_TOKEN_SPLIT.split(raw)),
        "subdomain_count": len(parts) - 2,
        "query_param_count": len(p.query.split("&")),
        "tld_length": len(parts[-1]),
        "path_length": len(p.path),
        "has_hyphen_in_domain": int("-" in netloc),
        "number_of_digits": digits,
        "tld_popularity": int(parts[-1] in POPULAR_TLDS),
        "suspicious_file_extension": int(p.path.lower().endswith(SUSPICIOUS_ENDINGS)),
        "domain_name_length": len(parts[-2]) if len(parts) > 1 else 0,
        "percentage_numeric_chars": (digits / len(raw) * 100) if raw else 0.0,
    }


def extract_frame(urls) -> pd.DataFrame:
    return pd.DataFrame([extract(u) for u in urls], columns=FEATURES)


# ---------------------------------------------------------------------------
# Set fitur TERKOREKSI: makna sama dengan 16 fitur di atas, tapi parsing URL
# diperbaiki, plus 2 fitur yang sebelumnya "bocor" ke fitur lain dibuat eksplisit.
#   - host diambil dari hostname yang benar (port & userinfo tidak ikut)
#   - sufiks dua tingkat (.co.uk, .co.id, ...) dikenali; subdomain minimal 0
#   - token = potongan alfanumerik yang tidak kosong
#   - query_param_count = 0 bila tidak ada query
#   - ekstensi mencurigakan harus diawali titik (".exe"), jadi "/cgi-bin" tidak ikut
#   - has_port dan no_scheme jadi fitur sendiri
# ---------------------------------------------------------------------------
FEATURES_FIXED = FEATURES + ["has_port", "no_scheme"]
SUSPICIOUS_EXT_FIXED = tuple("." + e for e in SUSPICIOUS_ENDINGS)
_ALNUM_SPLIT = re.compile(r"[^A-Za-z0-9]+")


def _fixed_parse(raw: str):
    no_scheme = not re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", raw)
    try:
        p = urlparse("http://" + raw if no_scheme else raw)
        host = (p.hostname or "").lower()
        try:
            port = p.port
        except ValueError:
            port = None
    except ValueError:
        p, host, port = _EMPTY, "", None
    return p, host, port, no_scheme


def extract_fixed(url: str) -> dict:
    raw = str(url).strip()
    p, host, port, no_scheme = _fixed_parse(raw)
    is_ip = bool(_IPV4.match(host))
    labels = [l for l in host.split(".") if l]
    if is_ip or not labels:
        suffix, main_label, sub_count = "", "", 0
    else:
        n_suffix = 2 if ".".join(labels[-2:]) in SECOND_LEVEL else 1
        suffix = ".".join(labels[-n_suffix:])
        main_label = labels[-n_suffix - 1] if len(labels) > n_suffix else ""
        sub_count = max(0, len(labels) - n_suffix - 1)
    digits = sum(ch.isdigit() for ch in raw)
    path = p.path or ""
    return {
        "url_length": len(raw),
        "has_ip_address": int(is_ip),
        "dot_count": raw.count("."),
        "https_flag": int(p.scheme.lower() == "https" and not no_scheme),
        "url_entropy": _entropy(raw),
        "token_count": len([t for t in _ALNUM_SPLIT.split(raw) if t]),
        "subdomain_count": sub_count,
        "query_param_count": len([q for q in p.query.split("&") if q]),
        "tld_length": len(suffix),
        "path_length": len(path),
        "has_hyphen_in_domain": int("-" in host),
        "number_of_digits": digits,
        "tld_popularity": int(suffix.split(".")[-1] in POPULAR_TLDS) if suffix else 0,
        "suspicious_file_extension": int(path.lower().endswith(SUSPICIOUS_EXT_FIXED)),
        "domain_name_length": len(main_label),
        "percentage_numeric_chars": (digits / len(raw) * 100) if raw else 0.0,
        "has_port": int(port is not None),
        "no_scheme": int(no_scheme),
    }


def extract_frame_fixed(urls) -> pd.DataFrame:
    return pd.DataFrame([extract_fixed(u) for u in urls], columns=FEATURES_FIXED)
