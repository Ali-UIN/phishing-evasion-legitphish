"""Cross-dataset test (Reviewer 2, comment 4): models trained on LegitPhish are
evaluated, without any retraining, on the PhiUSIIL Phishing URL dataset
(Prasad and Chandra, 2024; UCI Machine Learning Repository, CC BY 4.0).

    python lintas_dataset.py

PhiUSIIL is only used as test data. URLs whose canonical form also occurs in
LegitPhish are removed first. For every seed the four models are trained on the
LegitPhish training split in the three conditions of the main experiment (no
adversarial training, random, adaptive), then scored on
- all remaining PhiUSIIL URLs (recall, false positive rate), and
- the 65 disguising chains applied to a fixed random sample of 5,000 PhiUSIIL
  phishing URLs (adaptive evasion).
Scores go to cache/lintas_<seed>.npz, tables to results/revisi/r2_4_lintas_*.csv.
"""
from __future__ import annotations

import os
import random
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from analisis import agg, clean_metrics, evasion  # noqa: E402
from features_v2 import ALL_COLS, COLS_FIXED, canonical_key  # noqa: E402
from features import FEATURES_FIXED  # noqa: E402
from jalankan import (CACHE, MODELS, N_AUG_ADAPTIF, N_PROC, ROOT, TEST_SIZE, augment, extract_many,  # noqa: E402
                      fit_one, load_data, log, proba, safe_path, with_aug)
from perturb import generate_candidates  # noqa: E402
from perturb_v2 import candidate_chains, make_attacks  # noqa: E402

EXT = os.path.join(ROOT, "data", "eksternal", "PhiUSIIL_Phishing_URL_Dataset.csv")
OUT = os.path.join(ROOT, "results", "revisi")
SEEDS = (42, 7, 2025)
N_SAMPLE = 5000
CONDS = ["tanpa", "acak", "adaptif"]


def load_ext(legit_keys):
    raw = pd.read_csv(EXT, usecols=["URL", "label"], encoding="utf-8-sig").dropna()
    st = {"phiusiil_baris": int(len(raw))}
    raw["url"] = raw["URL"].astype(str).str.strip()
    raw["y"] = (raw["label"] == 0).astype(int)  # PhiUSIIL: 1 = legitimate, 0 = phishing
    raw["key"] = raw.url.map(canonical_key)
    raw = raw.drop_duplicates("key")
    st["phiusiil_setelah_dedup_kanonik"] = int(len(raw))
    sama = raw.key.isin(legit_keys)
    st["phiusiil_juga_ada_di_legitphish"] = int(sama.sum())
    ext = raw[~sama].reset_index(drop=True)
    st["phiusiil_dipakai"] = int(len(ext))
    st["phiusiil_phishing"] = int(ext.y.sum())
    st["phiusiil_sah"] = int((ext.y == 0).sum())
    return ext[["url", "y"]], st


def main():
    os.makedirs(CACHE, exist_ok=True)
    df, _ = load_data()
    ext, st = load_ext(set(df.key))
    log("PhiUSIIL:", st)
    y, urls = df.y.values, df.url.values.astype(object)
    c = COLS_FIXED
    samp = np.array(sorted(random.Random(0).sample(list(np.where(ext.y.values == 1)[0]), N_SAMPLE)))
    samp_urls = list(ext.url.values[samp])
    with Pool(N_PROC) as pool:
        X_all = extract_many(pool, urls)[:, c]
        X_ext_full = extract_many(pool, ext.url.values)
        X_ext = X_ext_full[:, c]
        for seed in SEEDS:
            out = os.path.join(CACHE, f"lintas_{seed}.npz")
            if os.path.exists(out):
                continue
            idx_tr, _ = train_test_split(np.arange(len(df)), test_size=TEST_SIZE, stratify=y, random_state=seed)
            tr_urls, y_tr = urls[idx_tr], y[idx_tr]
            train_phish = list(tr_urls[y_tr == 1])
            ATT = make_attacks([safe_path(u) for u in tr_urls[y_tr == 0]])
            CH = candidate_chains(ATT)
            src = random.Random(seed).sample(range(len(train_phish)), min(N_AUG_ADAPTIF, len(train_phish)))
            F_aug = extract_many(pool, augment(train_phish, ATT, list(ATT), 99))[:, c]
            F_ct = extract_many(pool, generate_candidates([train_phish[i] for i in src], ATT, CH, seed=99)[0])[:, c]
            F_ce = extract_many(pool, generate_candidates(samp_urls, ATT, CH, seed=4242)[0])[:, c]
            Xtr = X_all[idx_tr]
            rows0 = np.arange(len(src)) * len(CH)
            S = {}
            for name in MODELS:
                m0 = fit_one(name, Xtr, y_tr, seed)
                m1 = fit_one(name, *with_aug(Xtr, y_tr, F_aug), seed)
                pick = proba(m0, F_ct).reshape(len(src), len(CH)).argmin(1)
                m2 = fit_one(name, *with_aug(Xtr, y_tr, F_ct[rows0 + pick]), seed)
                for cond, m in zip(CONDS, (m0, m1, m2)):
                    S[f"{name}__{cond}__ext"] = proba(m, X_ext)
                    S[f"{name}__{cond}__cand"] = proba(m, F_ce)
                log(f"  seed {seed} {name}")
            np.savez_compressed(out, **S)

    # ---- tables -------------------------------------------------------------------------------------
    ye = ext.y.values
    col = {f: ALL_COLS.index(f) for f in FEATURES_FIXED}
    https = X_ext_full[:, col["https_flag"]] == 1
    path = X_ext_full[:, col["path_length"]]
    query = X_ext_full[:, col["query_param_count"]]
    beranda = https & (path <= 1) & (query == 0)
    legit = ye == 0
    st["phiusiil_porsi_sah_beranda_https"] = float(beranda[legit].mean())
    st["phiusiil_porsi_sah_ada_path"] = float((path > 1)[legit].mean())
    st["phiusiil_porsi_phishing_ada_path"] = float((path > 1)[ye == 1].mean())
    st["phiusiil_porsi_phishing_https"] = float(https[ye == 1].mean())
    rows = []
    for seed in SEEDS:
        S = np.load(os.path.join(CACHE, f"lintas_{seed}.npz"))
        for name in MODELS:
            for cond in CONDS:
                p = S[f"{name}__{cond}__ext"]
                cm = clean_metrics(ye, p)
                P = S[f"{name}__{cond}__cand"].reshape(N_SAMPLE, -1)
                det = p[samp] >= 0.5
                rows.append(dict(seed=seed, model=name, kondisi=cond, recall=cm["recall"], fpr=cm["fpr"], f1=cm["f1"],
                                 precision=cm["precision"], roc_auc=cm["roc_auc"],
                                 fpr_beranda_https=float((p[legit & beranda] >= 0.5).mean()),
                                 fpr_sah_lainnya=float((p[legit & ~beranda] >= 0.5).mean()),
                                 recall_sampel=float(det.mean()), evasion_adaptif=evasion(P, det)))
    d = pd.DataFrame(rows)
    d.to_csv(os.path.join(OUT, "r2_4_lintas_dataset_per_seed.csv"), index=False, float_format="%.6g")
    agg(d, ["model", "kondisi"]).to_csv(os.path.join(OUT, "r2_4_lintas_dataset.csv"), index=False, float_format="%.6g")
    pd.DataFrame([dict(keterangan=k, nilai=v) for k, v in st.items()]).to_csv(
        os.path.join(OUT, "r2_4_lintas_dataset_data.csv"), index=False)
    print(agg(d, ["model", "kondisi"])[["model", "kondisi", "recall_mean", "fpr_mean", "f1_mean", "evasion_adaptif_mean",
                                        "fpr_beranda_https_mean", "fpr_sah_lainnya_mean"]].round(4).to_string(index=False))
    print(st)


if __name__ == "__main__":
    main()
