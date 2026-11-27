"""Revision experiments, heavy part: trains the models and stores raw scores.

    python jalankan.py --kind acak   --seed 42      # stratified 80/20 split
    python jalankan.py --kind domain --seed 42      # split grouped by registered domain

Every stage writes one .npz file with model scores under revisi/cache/ and is
skipped when that file already exists. analisis.py turns the scores into the
tables under results/revisi/. The original notebook and its modules are not used
for training here and are left unchanged.

Stages of an "acak" run:
  utama_terkoreksi  4 models x {no AT, random AT, adaptive AT, adaptive AT on half
                    of the chain families}; scores on clean test data, blind
                    attacks, the 65 chains, 56 three-technique chains and the
                    extension chains; SHAP for LightGBM (seed 42)
  utama_replikasi   the same three training conditions with the replicated features
  val               models trained on 85% of the training data, the other 15% kept
                    as validation data for threshold tuning and calibration
  ablasi            feature ablation and host-based features (no AT)
  ngram             character n-gram TF-IDF + Logistic Regression baseline
  loao              leave-one-attack-out with LightGBM (seed 42)
A "domain" run only has utama_terkoreksi without the extension sets.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import sys
import time
import warnings
from multiprocessing import Pool
from types import SimpleNamespace
from urllib.parse import urlparse

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from lightgbm import LGBMClassifier  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.model_selection import GroupShuffleSplit, train_test_split  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from xgboost import XGBClassifier  # noqa: E402

from features import FEATURES_FIXED  # noqa: E402
from features_v2 import (ALL_COLS, COLS_FIXED, COLS_HOST, COLS_REPL, canonical_key,  # noqa: E402
                         extract_rows, registered_domain_v2)
from perturb import apply_attack, generate_candidates  # noqa: E402
from perturb_v2 import (EXT_ATTACKS, candidate_chains, extension_chains,  # noqa: E402
                        make_attacks, split_pair_families, triple_chains)

warnings.filterwarnings("ignore")
N_JOBS = int(os.environ.get("REVISI_NJOBS", "8"))
N_PROC = int(os.environ.get("REVISI_NPROC", "6"))
CACHE = os.path.join(HERE, "cache")
MODELS = ["LogReg", "RandomForest", "XGBoost", "LightGBM"]
TEST_SIZE = 0.2
N_AUG_ADAPTIF = 20000

_fx = {f: COLS_FIXED[i] for i, f in enumerate(FEATURES_FIXED)}
_MANIP = ["https_flag", "path_length", "query_param_count"]
_HOST_ONLY = ["has_ip_address", "subdomain_count", "tld_length", "has_hyphen_in_domain",
              "tld_popularity", "domain_name_length", "has_port"]
ABLATION = {
    "tanpa3": [_fx[f] for f in FEATURES_FIXED if f not in _MANIP],
    "tanpa3_host": [_fx[f] for f in FEATURES_FIXED if f not in _MANIP] + COLS_HOST,
    "penuh_host": COLS_FIXED + COLS_HOST,
    "host_saja": [_fx[f] for f in _HOST_ONLY] + COLS_HOST,
}


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def load_data(n=None):
    """Read LegitPhish, drop exact and canonical duplicates. Returns (df, counts)."""
    raw = pd.read_csv(sorted(glob.glob(os.path.join(ROOT, "data", "*.csv")))[0])
    df = raw[["URL", "ClassLabel"]].rename(columns={"URL": "url", "ClassLabel": "label_asli"})
    st = {"baris_berkas": int(len(df)) + 1}  # the published file has one more, incomplete line
    st["baris_berkas"] = int(len(df))
    df = df.dropna()
    st["baris_lengkap"] = int(len(df))
    df["url"] = df["url"].astype(str).str.strip()
    k = df.groupby("url")["label_asli"].nunique()
    st["url_label_ganda"] = int((k > 1).sum())
    df = df[~df.url.isin(k[k > 1].index)].drop_duplicates("url")
    st["setelah_dedup_persis"] = int(len(df))
    df["key"] = df.url.map(canonical_key)
    kk = df.groupby("key")["label_asli"].nunique()
    st["kunci_kanonik_label_ganda"] = int((kk > 1).sum())
    df = df[~df.key.isin(kk[kk > 1].index)]
    st["duplikat_kanonik_dibuang"] = int(df.key.duplicated().sum())
    st["kunci_longgar_bertabrakan"] = int(df.url.map(lambda u: canonical_key(u, True)).duplicated().sum())
    df = df.drop_duplicates("key").reset_index(drop=True)
    st["setelah_dedup_kanonik"] = int(len(df))
    df["y"] = (df.label_asli == 0).astype(int)
    df["domain"] = df.url.map(registered_domain_v2)
    if n:
        df = df.sample(n, random_state=0).reset_index(drop=True)
    return df, st


def make_models(seed):
    return {
        "LogReg": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
        "RandomForest": RandomForestClassifier(n_estimators=200, n_jobs=N_JOBS, random_state=seed),
        "XGBoost": XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1, tree_method="hist",
                                 eval_metric="logloss", n_jobs=N_JOBS, random_state=seed),
        "LightGBM": LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=63,
                                   random_state=seed, verbose=-1, n_jobs=N_JOBS),
    }


def fit_one(name, X, y, seed):
    m = make_models(seed)[name]
    m.fit(X, y)
    return m


def proba(m, X, chunk=400_000):
    out = np.empty(len(X), dtype=np.float32)
    for i in range(0, len(X), chunk):
        out[i:i + chunk] = m.predict_proba(X[i:i + chunk])[:, 1]
    return out


def safe_path(u):
    try:
        return urlparse(u if "://" in u else "http://" + u).path
    except ValueError:
        return ""


def augment(urls, attacks, names, seed):
    rng = random.Random(seed)
    return [attacks[rng.choice(names)](u, rng) for u in urls]


def extract_many(pool, urls, chunk=20000):
    urls = list(urls)
    if len(urls) <= chunk:
        return extract_rows(urls)
    return np.vstack(pool.map(extract_rows, [urls[i:i + chunk] for i in range(0, len(urls), chunk)]))


def with_aug(X, y, Xa):
    return np.vstack([X, Xa]), np.concatenate([y, np.ones(len(Xa), dtype=int)])


def record(S, tag, cond, name, m, sets):
    for sname, X in sets.items():
        S[f"{tag}__{name}__{cond}__{sname}"] = proba(m, X)
    if name == "LogReg":
        S[f"{tag}__LogReg__{cond}__coef"] = np.r_[m[-1].coef_[0], m[-1].intercept_]
        S[f"{tag}__LogReg__{cond}__scale"] = np.r_[m[0].mean_, m[0].scale_]


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------
def stage_utama(tag, c, ctx, S):
    F, seed = ctx.F, ctx.seed
    Xtr, ytr = ctx.X_all[ctx.idx_tr][:, c], ctx.y_tr
    sets = {"test": ctx.X_all[ctx.idx_te][:, c], "buta0": F["buta0"][:, c],
            "buta500": F["buta500"][:, c], "cand": F["cand_test"][:, c]}
    extra = tag == "terkoreksi" and ctx.kind == "acak"
    if extra:
        sets["tri"] = F["cand_tri"][:, c]
        sets["ext"] = F["cand_ext"][:, c]
    Xc = F["cand_train"][:, c]
    n_src, C = len(ctx.aug_src_idx), len(ctx.CH)
    rows0 = np.arange(n_src) * C
    Xa, ya = with_aug(Xtr, ytr, F["aug_acak"][:, c])
    lgb = {}
    for name in MODELS:
        t = time.perf_counter()
        m0 = fit_one(name, Xtr, ytr, seed)
        record(S, tag, "tanpa", name, m0, sets)
        m1 = fit_one(name, Xa, ya, seed)
        record(S, tag, "acak", name, m1, sets)
        p = proba(m0, Xc).reshape(n_src, C)
        pick = p.argmin(1)
        S[f"{tag}__{name}__picks"] = pick.astype(np.int16)
        m2 = fit_one(name, *with_aug(Xtr, ytr, Xc[rows0 + pick]), seed)
        record(S, tag, "adaptif", name, m2, sets)
        if extra:  # adaptive AT that only sees half of the two-technique chain families
            allowed = np.where(~ctx.heldout)[0]
            pick_h = allowed[p[:, allowed].argmin(1)]
            S[f"{tag}__{name}__picks_separuh"] = pick_h.astype(np.int16)
            m3 = fit_one(name, *with_aug(Xtr, ytr, Xc[rows0 + pick_h]), seed)
            record(S, tag, "separuh", name, m3, sets)
        if name == "LightGBM":
            lgb = {"tanpa": m0, "acak": m1, "adaptif": m2}
        log(f"  {tag} {name}: {time.perf_counter() - t:.0f}s")
    if extra and seed == 42:
        import shap
        idx = np.random.RandomState(42).choice(len(ctx.idx_te), min(2000, len(ctx.idx_te)), replace=False)
        Xs = sets["test"][idx]
        for cond, m in lgb.items():
            sv = shap.TreeExplainer(m).shap_values(Xs)
            sv = np.asarray(sv[1] if isinstance(sv, list) else sv)
            if sv.ndim == 3:
                sv = sv[:, :, 1]
            S[f"shap__{cond}"] = np.abs(sv).mean(0)


def stage_val(c, ctx, S):
    """Same three training conditions on 85% of the training data; the remaining
    15% is a validation set that no model sees during training."""
    F, seed = ctx.F, ctx.seed
    pos = np.arange(len(ctx.idx_tr))
    sub, val = train_test_split(pos, test_size=0.15, stratify=ctx.y_tr, random_state=seed)
    in_sub = np.zeros(len(pos), dtype=bool)
    in_sub[sub] = True
    Xfull = ctx.X_all[ctx.idx_tr][:, c]
    Xs, ys = Xfull[sub], ctx.y_tr[sub]
    pos_phish = np.where(ctx.y_tr == 1)[0]
    C = len(ctx.CH)
    src_keep = in_sub[pos_phish[ctx.aug_src_idx]]
    Xc = F["cand_train"][:, c].reshape(len(ctx.aug_src_idx), C, len(c))[src_keep].reshape(-1, len(c))
    n_src = int(src_keep.sum())
    rows0 = np.arange(n_src) * C
    sets = {"val": Xfull[val], "test": ctx.X_all[ctx.idx_te][:, c], "cand": F["cand_test"][:, c]}
    S["val__y"] = ctx.y_tr[val]
    Xa, ya = with_aug(Xs, ys, F["aug_acak"][:, c][in_sub[pos_phish]])
    for name in MODELS:
        m0 = fit_one(name, Xs, ys, seed)
        record(S, "val", "tanpa", name, m0, sets)
        record(S, "val", "acak", name, fit_one(name, Xa, ya, seed), sets)
        pick = proba(m0, Xc).reshape(n_src, C).argmin(1)
        record(S, "val", "adaptif", name, fit_one(name, *with_aug(Xs, ys, Xc[rows0 + pick]), seed), sets)
        log(f"  val {name}")


def stage_ablasi(ctx, S):
    F, seed = ctx.F, ctx.seed
    for cfg, c in ABLATION.items():
        Xtr = ctx.X_all[ctx.idx_tr][:, c]
        sets = {"test": ctx.X_all[ctx.idx_te][:, c], "buta0": F["buta0"][:, c], "cand": F["cand_test"][:, c]}
        for name in MODELS:
            record(S, f"abl_{cfg}", "tanpa", name, fit_one(name, Xtr, ctx.y_tr, seed), sets)
        log(f"  ablasi {cfg}")


def stage_ngram(ctx, S):
    """Sequence-based baseline without hand-made features: character 1-3-gram
    TF-IDF of the raw URL string + Logistic Regression."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    U = ctx.U

    def fit(urls, y):
        vec = TfidfVectorizer(analyzer="char", ngram_range=(1, 3), lowercase=False, min_df=2,
                              sublinear_tf=True, dtype=np.float32)
        clf = LogisticRegression(solver="liblinear", max_iter=1000)
        clf.fit(vec.fit_transform(urls), y)
        return vec, clf

    def score(model, urls, chunk=100_000):
        vec, clf = model
        out = np.empty(len(urls), dtype=np.float32)
        for i in range(0, len(urls), chunk):
            out[i:i + chunk] = clf.predict_proba(vec.transform(urls[i:i + chunk]))[:, 1]
        return out

    tr, ytr = list(ctx.tr_urls), ctx.y_tr
    sets = {"test": list(ctx.te_urls), "buta0": U["buta0"], "buta500": U["buta500"], "cand": U["cand_test"]}
    n_src, C = len(ctx.aug_src_idx), len(ctx.CH)
    m0 = fit(tr, ytr)
    S["ngram__n_fitur"] = np.array([len(m0[0].vocabulary_)])
    pick = score(m0, U["cand_train"]).reshape(n_src, C).argmin(1)
    S["ngram__NGram__picks"] = pick.astype(np.int16)
    picked = [U["cand_train"][i * C + j] for i, j in enumerate(pick)]
    ones = lambda k: np.concatenate([ytr, np.ones(k, dtype=int)])  # noqa: E731
    models = {"tanpa": m0,
              "acak": fit(tr + U["aug_acak"], ones(len(U["aug_acak"]))),
              "adaptif": fit(tr + picked, ones(len(picked)))}
    for cond, m in models.items():
        for sname, urls in sets.items():
            S[f"ngram__NGram__{cond}__{sname}"] = score(m, urls)
        log(f"  ngram {cond}")


def stage_loao(c, ctx, S):
    F, seed = ctx.F, ctx.seed
    Xtr, ytr = ctx.X_all[ctx.idx_tr][:, c], ctx.y_tr
    Xtp = ctx.X_all[ctx.idx_te][:, c][ctx.y_te == 1]
    for held in ctx.base_names:
        m = fit_one("LightGBM", *with_aug(Xtr, ytr, F[f"loao_aug__{held}"][:, c]), seed)
        S[f"loao__{held}__clean"] = proba(m, Xtp)
        S[f"loao__{held}__adv"] = proba(m, F[f"loao_test__{held}"][:, c])
    log("  loao selesai")


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=["acak", "domain"], default="acak")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n", type=int, default=None, help="subsample for a smoke test")
    ap.add_argument("--stages", default="")
    a = ap.parse_args()
    kind, seed = a.kind, a.seed
    os.makedirs(CACHE, exist_ok=True)
    out = os.path.join(CACHE, f"{kind}_{seed}" + (f"_n{a.n}" if a.n else ""))
    stages = ["utama_terkoreksi", "utama_replikasi"] if kind == "domain" else \
        ["utama_terkoreksi", "utama_replikasi", "val", "ablasi", "ngram"] + (["loao"] if seed == 42 else [])
    if a.stages:
        stages = [s for s in stages if s in a.stages.split(",")]
    todo = [s for s in stages if not os.path.exists(f"{out}__{s}.npz")]
    if not todo:
        log("semua tahap sudah ada:", out)
        return
    log(f"mulai {kind} seed {seed} | tahap: {todo}")

    df, st = load_data(a.n)
    y = df.y.values
    urls = df.url.values.astype(object)
    if kind == "acak":
        idx_tr, idx_te = train_test_split(np.arange(len(df)), test_size=TEST_SIZE, stratify=y, random_state=seed)
    else:
        gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=seed)
        idx_tr, idx_te = next(gss.split(urls, y, groups=df.domain.values))
    ctx = SimpleNamespace(kind=kind, seed=seed, idx_tr=idx_tr, idx_te=idx_te, y_tr=y[idx_tr], y_te=y[idx_te],
                          tr_urls=urls[idx_tr], te_urls=urls[idx_te])
    train_phish = list(ctx.tr_urls[ctx.y_tr == 1])
    test_phish = list(ctx.te_urls[ctx.y_te == 1])
    ATT = make_attacks([safe_path(u) for u in ctx.tr_urls[ctx.y_tr == 0]])  # legit paths: training data only
    ctx.CH = candidate_chains(ATT)
    ctx.base_names = [k for k in ATT if k != "combo"]
    held_pairs = split_pair_families(seed)
    ctx.heldout = np.array([len(ch) == 2 and frozenset(ch) in held_pairs for ch in ctx.CH])
    tri, ext = triple_chains(), extension_chains()

    t = time.perf_counter()
    U = {}
    U["buta0"] = sum((apply_attack(test_phish, fn, seed=1000 + i) for i, fn in enumerate(ATT.values())), [])
    U["buta500"] = sum((apply_attack(test_phish, fn, seed=1500 + i) for i, fn in enumerate(ATT.values())), [])
    U["aug_acak"] = augment(train_phish, ATT, list(ATT), 99)
    U["cand_test"] = generate_candidates(test_phish, ATT, ctx.CH, seed=4242)[0]
    ctx.aug_src_idx = np.array(random.Random(seed).sample(range(len(train_phish)), min(N_AUG_ADAPTIF, len(train_phish))))
    U["cand_train"] = generate_candidates([train_phish[i] for i in ctx.aug_src_idx], ATT, ctx.CH, seed=99)[0]
    if kind == "acak":
        U["cand_tri"] = generate_candidates(test_phish, ATT, tri, seed=4343)[0]
        U["cand_ext"] = generate_candidates(test_phish, {**ATT, **EXT_ATTACKS}, ext, seed=4444)[0]
        if "loao" in todo:
            for held in ctx.base_names:
                U[f"loao_aug__{held}"] = augment(train_phish, ATT, [b for b in ctx.base_names if b != held], 123)
                U[f"loao_test__{held}"] = apply_attack(test_phish, ATT[held], seed=777)
    log(f"URL samaran dibuat: {sum(len(v) for v in U.values()):,} ({time.perf_counter() - t:.0f}s)")

    t = time.perf_counter()
    with Pool(N_PROC) as pool:
        ctx.X_all = extract_many(pool, urls)
        ctx.F = {k: extract_many(pool, v) for k, v in U.items()}
    ctx.U = U
    log(f"ekstraksi fitur selesai ({time.perf_counter() - t:.0f}s)")

    meta = dict(st, kind=kind, seed=seed, n=len(df), n_latih=len(idx_tr), n_uji=len(idx_te),
                latih_phishing=int(ctx.y_tr.sum()), uji_phishing=int(ctx.y_te.sum()),
                domain_uji=int(df.domain.iloc[idx_te].nunique()),
                contoh={k: v[:3] for k, v in U.items() if not k.startswith("loao")})
    for cls, nama in [(0, "sah"), (1, "phishing")]:
        vc = df.loc[df.y == cls, "domain"].value_counts()
        meta[f"domain_unik_{nama}"] = int(len(vc))
        meta[f"porsi_top10_{nama}"] = float(vc.head(10).sum() / vc.sum())
    with open(out + "__meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)
    pd.DataFrame({"url": ctx.te_urls, "y": ctx.y_te}).to_csv(out + "__uji.csv", index=False)

    common = dict(y_te=ctx.y_te, X_te=ctx.X_all[idx_te], cols=np.array(ALL_COLS),
                  chain_names=np.array([" + ".join(ch) for ch in ctx.CH]), heldout=ctx.heldout,
                  tri_names=np.array([" + ".join(ch) for ch in tri]),
                  ext_names=np.array([" + ".join(ch) for ch in ext]),
                  attack_names=np.array(list(ATT)))
    for stage in todo:
        t = time.perf_counter()
        S = {}
        if stage == "utama_terkoreksi":
            S.update(common)
            stage_utama("terkoreksi", COLS_FIXED, ctx, S)
        elif stage == "utama_replikasi":
            stage_utama("replikasi", COLS_REPL, ctx, S)
        elif stage == "val":
            stage_val(COLS_FIXED, ctx, S)
        elif stage == "ablasi":
            stage_ablasi(ctx, S)
        elif stage == "ngram":
            stage_ngram(ctx, S)
        elif stage == "loao":
            stage_loao(COLS_FIXED, ctx, S)
        np.savez_compressed(f"{out}__{stage}.npz", **S)
        log(f"tahap {stage} disimpan ({time.perf_counter() - t:.0f}s)")
    log("selesai")


if __name__ == "__main__":
    main()
