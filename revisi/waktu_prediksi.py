"""Prediction time per URL for Table 5 (features already extracted, as in the
original notebook). Three seeds, single run each; writes
results/revisi/tabel5_waktu_prediksi.csv.

    python waktu_prediksi.py
"""
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from features_v2 import COLS_FIXED  # noqa: E402
from jalankan import MODELS, N_PROC, ROOT, extract_many, load_data, make_models  # noqa: E402


def main():
    df, _ = load_data()
    y, urls = df.y.values, df.url.values.astype(object)
    with Pool(N_PROC) as pool:
        X = extract_many(pool, urls)[:, COLS_FIXED]
    rows = []
    for seed in (42, 7, 2025):
        tr, te = train_test_split(np.arange(len(df)), test_size=0.2, stratify=y, random_state=seed)
        for name in MODELS:
            m = make_models(seed)[name].fit(X[tr], y[tr])
            t = time.perf_counter()
            m.predict_proba(X[te])
            rows.append(dict(seed=seed, model=name, ms_per_url=(time.perf_counter() - t) / len(te) * 1000))
        vec = TfidfVectorizer(analyzer="char", ngram_range=(1, 3), lowercase=False, min_df=2,
                              sublinear_tf=True, dtype=np.float32)
        clf = LogisticRegression(solver="liblinear", max_iter=1000).fit(vec.fit_transform(list(urls[tr])), y[tr])
        t = time.perf_counter()
        clf.predict_proba(vec.transform(list(urls[te])))  # includes turning the URL into n-grams
        rows.append(dict(seed=seed, model="NGram", ms_per_url=(time.perf_counter() - t) / len(te) * 1000))
    d = pd.DataFrame(rows)
    out = d.groupby("model", sort=False)["ms_per_url"].agg(["mean", "std"]).reset_index()
    out.columns = ["model", "ms_per_url_mean", "ms_per_url_sd"]
    out.to_csv(os.path.join(ROOT, "results", "revisi", "tabel5_waktu_prediksi.csv"), index=False, float_format="%.6g")
    print(out)


if __name__ == "__main__":
    main()
