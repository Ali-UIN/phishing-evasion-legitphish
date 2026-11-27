"""Revision experiments, light part: turns the scores stored by jalankan.py into
the tables under results/revisi/. Every number in the revised manuscript comes
from one of these CSV files.

    python analisis.py            # full runs (seeds 42, 7, 2025)
    python analisis.py --suf _n8000 --seeds 42     # smoke-test cache
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import numpy as np
import pandas as pd
from scipy.special import gammaln
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
from features import FEATURES, FEATURES_FIXED, registered_domain  # noqa: E402
from features_v2 import ALL_COLS, HOST_FEATURES  # noqa: E402

CACHE = os.path.join(HERE, "cache")
OUT = os.path.join(ROOT, "results", "revisi")
MODELS = ["LogReg", "RandomForest", "XGBoost", "LightGBM"]
CONDS = ["tanpa", "acak", "adaptif"]
STAGE = {"terkoreksi": "utama_terkoreksi", "replikasi": "utama_replikasi", "val": "val", "ngram": "ngram"}
BASE_ATTACKS = ["force_https", "root_landing", "typosquat", "homoglyph", "brand_subdomain",
                "at_trick", "strip_query", "mimic_legit_path", "shortener"]
# Operational feasibility (Reviewer 2, comment 1)
TIER1 = {"typosquat", "homoglyph", "brand_subdomain", "shortener"}
TIERS = {
    "T1_tanpa_ubah_server": TIER1,
    "T1_plus_https": TIER1 | {"force_https"},
    "semua_kecuali_at_trick": set(BASE_ATTACKS) - {"at_trick"},
    "semua_65_rantai": set(BASE_ATTACKS),
}


class Run:
    def __init__(self, kind, seed, suf=""):
        self.kind, self.seed = kind, seed
        self.base = os.path.join(CACHE, f"{kind}_{seed}{suf}")
        self._z = {}
        u = self.z("utama_terkoreksi")
        self.y = u["y_te"]
        self.tp = self.y == 1
        self.n_tp = int(self.tp.sum())
        self.chains = [str(s) for s in u["chain_names"]]
        self.attacks = [str(s) for s in u["attack_names"]]
        self.heldout = u["heldout"]
        self.meta = json.load(open(self.base + "__meta.json", encoding="utf-8"))

    def has(self, stage):
        return os.path.exists(f"{self.base}__{stage}.npz")

    def z(self, stage):
        if stage not in self._z:
            self._z[stage] = np.load(f"{self.base}__{stage}.npz")
        return self._z[stage]

    def p(self, tag, model, cond, sname):
        stage = STAGE.get(tag, "ablasi")
        return self.z(stage)[f"{tag}__{model}__{cond}__{sname}"]

    def det(self, tag, model, cond, thr=0.5):
        return self.p(tag, model, cond, "test")[self.tp] >= thr

    def cand(self, tag, model, cond, sname="cand"):
        return self.p(tag, model, cond, sname).reshape(self.n_tp, -1)


def clean_metrics(y, p, thr=0.5):
    pred = p >= thr
    tp = int((pred & (y == 1)).sum()); fp = int((pred & (y == 0)).sum())
    fn = int((~pred & (y == 1)).sum()); tn = int((~pred & (y == 0)).sum())
    prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
    return dict(accuracy=(tp + tn) / len(y), precision=prec, recall=rec,
                f1=2 * prec * rec / max(prec + rec, 1e-12), roc_auc=roc_auc_score(y, p),
                fpr=fp / max(fp + tn, 1))


def evasion(P, det, thr=0.5):
    return float(((P.min(1) < thr) & det).sum() / max(det.sum(), 1))


def agg(df, by, cols=None):
    cols = cols or [c for c in df.columns if c not in by + ["seed"]]
    g = df.groupby(by, sort=False)[cols].agg(["mean", "std"])
    g.columns = [f"{c}_{'mean' if s == 'mean' else 'sd'}" for c, s in g.columns]
    g["n_seed"] = df.groupby(by, sort=False).size()
    return g.reset_index()


def save(df, name):
    df.to_csv(os.path.join(OUT, name), index=False, float_format="%.6g")
    print(f"  {name}: {len(df)} baris")


def blind_table(r, tag, model, cond):
    sname = "buta0" if cond == "tanpa" else "buta500"
    B = r.p(tag, model, cond, sname).reshape(len(r.attacks), r.n_tp)
    det = r.det(tag, model, cond)
    return ((B < 0.5) & det).sum(1) / max(det.sum(), 1)


def tags_models(r):
    out = [("terkoreksi", m) for m in MODELS]
    if r.has("utama_replikasi"):
        out += [("replikasi", m) for m in MODELS]
    if r.has("ngram"):
        out += [("ngram", "NGram")]
    return out


# ---------------------------------------------------------------------------
def t_utama(runs):
    """Tables 5, 6, 7, 8, 9, 11, 12 of the manuscript (mean and SD over seeds)."""
    bersih, buta, adapt, menang = [], [], [], []
    for r in runs:
        for tag, m in tags_models(r):
            for cond in CONDS:
                row = dict(split=r.kind, seed=r.seed, set_fitur=tag, model=m, kondisi=cond)
                bersih.append(dict(row, **clean_metrics(r.y, r.p(tag, m, cond, "test"))))
                ev = blind_table(r, tag, m, cond)
                for a, v in zip(r.attacks, ev):
                    buta.append(dict(row, teknik=a, evasion=float(v)))
                buta.append(dict(row, teknik="RATA-RATA", evasion=float(ev.mean())))
                P, det = r.cand(tag, m, cond), r.det(tag, m, cond)
                best = P.min(1)
                adapt.append(dict(row, evasion_adaptif=evasion(P, det),
                                  median_skor_terbaik=float(np.median(best[det])) if det.any() else np.nan))
                ok = (best < 0.5) & det
                cnt = np.bincount(P.argmin(1)[ok], minlength=P.shape[1])
                for j, c in enumerate(cnt):
                    menang.append(dict(row, rantai=r.chains[j], jumlah=int(c), porsi=c / max(ok.sum(), 1)))
    bersih, buta, adapt, menang = map(pd.DataFrame, (bersih, buta, adapt, menang))
    by = ["split", "set_fitur", "model", "kondisi"]
    save(bersih, "bersih_per_seed.csv")
    save(agg(bersih, by), "tabel9_bersih_tiga_kondisi.csv")
    save(agg(bersih[bersih.kondisi == "tanpa"], by), "tabel5_baseline.csv")
    save(agg(buta, by + ["teknik"]), "tabel6_evasion_buta.csv")
    save(adapt, "tabel7_evasion_adaptif_per_seed.csv")
    save(agg(adapt, by), "tabel7_evasion_adaptif.csv")
    mm = agg(menang, by + ["rantai"], ["jumlah", "porsi"])
    s42 = menang[menang.seed == 42].rename(columns={"jumlah": "jumlah_seed42"})[by + ["rantai", "jumlah_seed42"]]
    save(mm.merge(s42, on=by + ["rantai"], how="left"), "tabel8_rantai_menang.csv")

    # Table 12: replicated against corrected features (random split unless stated)
    a = agg(bersih[(bersih.kondisi == "tanpa")], ["split", "set_fitur", "model"], ["f1", "fpr"])
    rows = []
    for tag in ["replikasi", "terkoreksi"]:
        for m in MODELS:
            def one(df, col, **kw):
                q = df[(df.set_fitur == tag) & (df.model == m)]
                for k, v in kw.items():
                    q = q[q[k] == v]
                return (q[col].mean(), q[col].std()) if len(q) else (np.nan, np.nan)
            row = dict(model=m, set_fitur=tag)
            for name, df, col, kw in [
                ("f1", bersih, "f1", dict(split="acak", kondisi="tanpa")),
                ("fpr", bersih, "fpr", dict(split="acak", kondisi="tanpa")),
                ("evasion_buta", buta, "evasion", dict(split="acak", kondisi="tanpa", teknik="RATA-RATA")),
                ("adaptif_tanpa", adapt, "evasion_adaptif", dict(split="acak", kondisi="tanpa")),
                ("adaptif_at_adaptif", adapt, "evasion_adaptif", dict(split="acak", kondisi="adaptif")),
                ("f1_split_domain", bersih, "f1", dict(split="domain", kondisi="tanpa")),
                ("fpr_split_domain", bersih, "fpr", dict(split="domain", kondisi="tanpa")),
            ]:
                row[name + "_mean"], row[name + "_sd"] = one(df, col, **kw)
            rows.append(row)
    save(pd.DataFrame(rows), "tabel12_replikasi_vs_terkoreksi.csv")
    return bersih, buta, adapt


def t_heldout(runs):
    """R1-3a: adversarial training that only sees half of the chain families."""
    rows = []
    for r in runs:
        if r.kind != "acak":
            continue
        z = r.z("utama_terkoreksi")
        held = r.heldout
        single = np.array([" + " not in c for c in r.chains])
        for m in MODELS:
            for cond in CONDS + ["separuh"]:
                if f"terkoreksi__{m}__{cond}__cand" not in z.files:
                    continue
                P, T, det = r.cand("terkoreksi", m, cond), r.cand("terkoreksi", m, cond, "tri"), r.det("terkoreksi", m, cond)
                rows.append(dict(seed=r.seed, model=m, kondisi=cond,
                                 rantai_dilatih_37=evasion(P[:, ~held], det),
                                 pasangan_disisihkan_28=evasion(P[:, held], det),
                                 pasangan_semua_56=evasion(P[:, ~single], det),
                                 tiga_teknik_56=evasion(T, det),
                                 semua_65=evasion(P, det),
                                 gabungan_121=evasion(np.hstack([P, T]), det)))
        fam = sorted({" / ".join(sorted(c.split(" + "))) for c, h in zip(r.chains, held) if h})
        rows_f = getattr(t_heldout, "fam", [])
        rows_f.append(dict(seed=r.seed, keluarga_disisihkan="; ".join(fam)))
        t_heldout.fam = rows_f
    df = pd.DataFrame(rows)
    save(df, "r1_3_rantai_heldout_per_seed.csv")
    save(agg(df, ["model", "kondisi"]), "r1_3_rantai_heldout.csv")
    save(pd.DataFrame(t_heldout.fam), "r1_3_keluarga_disisihkan.csv")


def t_ambang(runs):
    """R1-3b: thresholds tuned on a validation split, FPR at fixed recall, calibration."""
    rows = []
    for r in runs:
        if r.kind != "acak" or not r.has("val"):
            continue
        z = r.z("val")
        yv = z["val__y"]
        for m in MODELS:
            for cond in CONDS:
                pv, pt = z[f"val__{m}__{cond}__val"], z[f"val__{m}__{cond}__test"]
                P = z[f"val__{m}__{cond}__cand"].reshape(r.n_tp, -1)
                legit = np.sort(pv[yv == 0])[::-1]
                thr = {"ambang_0.5": 0.5,
                       "recall_val_99%": float(np.quantile(pv[yv == 1], 0.01, method="lower")),
                       "recall_val_99.9%": float(np.quantile(pv[yv == 1], 0.001, method="lower"))}
                for a in (0.001, 0.002, 0.005):
                    thr[f"fpr_val_{a * 100:g}%"] = float(np.nextafter(legit[int(a * len(legit))], np.float32(2)))
                for titik, t in thr.items():
                    det = pt[r.tp] >= t
                    cm = clean_metrics(r.y, pt, t)
                    rows.append(dict(seed=r.seed, model=m, kondisi=cond, titik=titik, ambang=t, recall=cm["recall"],
                                     fpr=cm["fpr"], f1=cm["f1"], evasion_adaptif=evasion(P, det, t)))
                iso = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip").fit(pv, yv)
                ct = iso.predict(pt)
                cP = iso.predict(P.ravel()).reshape(P.shape)
                cm = clean_metrics(r.y, ct)
                rows.append(dict(seed=r.seed, model=m, kondisi=cond, titik="isotonik_0.5", ambang=0.5,
                                 recall=cm["recall"], fpr=cm["fpr"], f1=cm["f1"],
                                 evasion_adaptif=evasion(cP, ct[r.tp] >= 0.5),
                                 brier_mentah=brier_score_loss(r.y, pt), brier_isotonik=brier_score_loss(r.y, ct),
                                 average_precision=average_precision_score(r.y, pt)))
    df = pd.DataFrame(rows)
    if len(df):
        save(df, "r1_3_ambang_per_seed.csv")
        save(agg(df, ["model", "kondisi", "titik"]), "r1_3_ambang.csv")


def t_logreg(runs):
    """R1-4: Logistic Regression coefficients and the mix of adaptive training samples."""
    coef, mix, per = [], [], []
    for r in runs:
        if r.kind != "acak":
            continue
        for tag, feats in [("terkoreksi", FEATURES_FIXED), ("replikasi", FEATURES)]:
            if not r.has(STAGE[tag]):
                continue
            z = r.z(STAGE[tag])
            for cond in CONDS:
                w = z[f"{tag}__LogReg__{cond}__coef"]
                sc = z[f"{tag}__LogReg__{cond}__scale"]
                for i, f in enumerate(feats + ["(intercept)"]):
                    coef.append(dict(seed=r.seed, set_fitur=tag, fitur=f, kondisi=cond, koefisien=float(w[i]),
                                     rerata_latih=float(sc[i]) if i < len(feats) else np.nan,
                                     simpangan_latih=float(sc[len(feats) + i]) if i < len(feats) else np.nan))
                P, det = r.cand(tag, "LogReg", cond), r.det(tag, "LogReg", cond)
                ok = (P < 0.5) & det[:, None]
                for j, c in enumerate(r.chains):
                    per.append(dict(seed=r.seed, set_fitur=tag, kondisi=cond, rantai=c,
                                    lolos=float(ok[:, j].sum() / max(det.sum(), 1))))
            for m in MODELS:
                pk = z[f"{tag}__{m}__picks"]
                cnt = np.bincount(pk, minlength=len(r.chains)) / len(pk)
                for j, c in enumerate(r.chains):
                    mix.append(dict(seed=r.seed, set_fitur=tag, model=m, rantai=c, porsi=float(cnt[j])))
    coef, mix, per = map(pd.DataFrame, (coef, mix, per))
    c = agg(coef, ["set_fitur", "fitur", "kondisi"], ["koefisien", "rerata_latih", "simpangan_latih"])
    save(c, "r1_4_koefisien_logreg.csv")
    wide = c.pivot(index=["set_fitur", "fitur"], columns="kondisi", values="koefisien_mean")[CONDS].reset_index()
    save(wide, "r1_4_koefisien_logreg_lebar.csv")
    save(agg(mix, ["set_fitur", "model", "rantai"]).sort_values(["set_fitur", "model", "porsi_mean"],
                                                               ascending=[True, True, False]), "r1_4_campuran_rantai.csv")
    pw = agg(per, ["set_fitur", "kondisi", "rantai"]).pivot(index=["set_fitur", "rantai"], columns="kondisi",
                                                           values="lolos_mean")[CONDS].reset_index()
    save(pw, "r1_4_logreg_lolos_per_rantai.csv")


def t_tiers(runs):
    """R2-1: adaptive evasion when the attacker is limited to a feasibility tier."""
    rows = []
    for r in runs:
        if r.kind != "acak":
            continue
        tech = [set(c.split(" + ")) for c in r.chains]
        for m in MODELS:
            for cond in CONDS:
                P, det = r.cand("terkoreksi", m, cond), r.det("terkoreksi", m, cond)
                for tier, allowed in TIERS.items():
                    mask = np.array([t <= allowed for t in tech])
                    rows.append(dict(seed=r.seed, model=m, kondisi=cond, tingkat=tier, n_rantai=int(mask.sum()),
                                     evasion_adaptif=evasion(P[:, mask], det)))
    df = pd.DataFrame(rows)
    save(agg(df, ["model", "kondisi", "tingkat", "n_rantai"]), "r2_1_tingkat_kelayakan.csv")


def _p_within(s, n, k):
    """P(at least one of s successful chains among k drawn without replacement from n)."""
    s = np.asarray(s, dtype=float)
    out = np.ones_like(s)
    ok = n - s >= k
    out[ok] = 1 - np.exp(gammaln(n - s[ok] + 1) - gammaln(n - s[ok] - k + 1) - gammaln(n + 1) + gammaln(n - k + 1))
    out[s == 0] = 0.0
    return out


def t_budget(runs):
    """R2-2: query budget and a hard-label attacker that stops at the first success."""
    KS = [1, 5, 10, 20, 40, 65]
    rows, hard = [], []
    for r in runs:
        if r.kind != "acak":
            continue
        for m in MODELS:
            for cond in CONDS:
                P, det = r.cand("terkoreksi", m, cond), r.det("terkoreksi", m, cond)
                ok = (P < 0.5)[det]
                n, C = ok.shape
                s = ok.sum(1)
                # informed order: chains ranked by success frequency on the other half of the URLs
                half = np.random.RandomState(r.seed).permutation(n) < n // 2
                pos = np.full(n, C + 1)
                for a, b in [(half, ~half), (~half, half)]:
                    order = np.argsort(-ok[a].sum(0), kind="stable")
                    o = ok[b][:, order]
                    pos[b] = np.where(o.any(1), o.argmax(1) + 1, C + 1)
                for k in KS:
                    rows.append(dict(seed=r.seed, model=m, kondisi=cond, K=k,
                                     urutan_acak=float(_p_within(s, C, k).mean()),
                                     urutan_terinformasi=float((pos <= k).mean())))
                succ = s > 0
                hard.append(dict(seed=r.seed, model=m, kondisi=cond, evasion_hard_label=float(succ.mean()),
                                 kueri_rata2_sampai_lolos_acak=float(((C + 1) / (s[succ] + 1)).mean()) if succ.any() else np.nan,
                                 kueri_rata2_sampai_lolos_terinformasi=float(pos[succ].mean()) if succ.any() else np.nan,
                                 kueri_median_sampai_lolos_terinformasi=float(np.median(pos[succ])) if succ.any() else np.nan,
                                 kueri_rata2_per_url_terinformasi=float(np.minimum(pos, C).mean())))
    save(agg(pd.DataFrame(rows), ["model", "kondisi", "K"]), "r2_2_anggaran_kueri.csv")
    save(agg(pd.DataFrame(hard), ["model", "kondisi"]), "r2_2_hard_label.csv")


def t_ablasi(runs):
    """R2-3: feature ablation and host-based features, models without adversarial training."""
    rows = []
    for r in runs:
        if r.kind != "acak" or not r.has("ablasi"):
            continue
        for cfg in ["penuh", "tanpa3", "tanpa3_host", "penuh_host", "host_saja"]:
            tag = "terkoreksi" if cfg == "penuh" else f"abl_{cfg}"
            for m in MODELS:
                cm = clean_metrics(r.y, r.p(tag, m, "tanpa", "test"))
                det = r.det(tag, m, "tanpa")
                B = r.p(tag, m, "tanpa", "buta0").reshape(len(r.attacks), r.n_tp)
                rows.append(dict(seed=r.seed, konfigurasi=cfg, model=m, f1=cm["f1"], recall=cm["recall"], fpr=cm["fpr"],
                                 evasion_buta=float((((B < 0.5) & det).sum(1) / max(det.sum(), 1)).mean()),
                                 evasion_adaptif=evasion(r.cand(tag, m, "tanpa"), det)))
    df = pd.DataFrame(rows)
    if len(df):
        save(agg(df, ["konfigurasi", "model"]), "r2_3_ablasi.csv")


def t_fp(runs):
    """R2-4a: which legitimate URLs become false positives after adversarial training."""
    col = {f: ALL_COLS.index(f) for f in FEATURES_FIXED}
    prof, cat, contoh = [], [], []
    for r in runs:
        X = r.z("utama_terkoreksi")["X_te"]
        legit = ~r.tp
        g = lambda f: X[:, col[f]]  # noqa: E731
        https, path, query = g("https_flag") == 1, g("path_length"), g("query_param_count")
        kategori = np.select(
            [https & (path <= 1) & (query == 0), ~https & (path <= 1) & (query == 0),
             https & (path > 1) & (path <= 15), https & (path > 15)],
            ["beranda_https", "beranda_tanpa_https", "https_path_pendek_2_15", "https_path_panjang"],
            default="lainnya_tanpa_https")
        urls = pd.read_csv(r.base + "__uji.csv")["url"].values
        for m in MODELS[1:]:
            fp = {c: (r.p("terkoreksi", m, c, "test") >= 0.5) & legit for c in CONDS}
            groups = {"semua_url_sah_uji": legit, "fp_tanpa": fp["tanpa"],
                      "fp_baru_acak": fp["acak"] & ~fp["tanpa"], "fp_baru_adaptif": fp["adaptif"] & ~fp["tanpa"]}
            for gname, mask in groups.items():
                if not mask.any():
                    continue
                prof.append(dict(split=r.kind, seed=r.seed, model=m, kelompok=gname, n=int(mask.sum()),
                                 porsi_https=float(https[mask].mean()),
                                 porsi_path_maks1=float((path[mask] <= 1).mean()),
                                 porsi_tanpa_query=float((query[mask] == 0).mean()),
                                 porsi_tanpa_subdomain=float((g("subdomain_count")[mask] == 0).mean()),
                                 porsi_tld_populer=float(g("tld_popularity")[mask].mean()),
                                 porsi_ada_digit=float((g("number_of_digits")[mask] > 0).mean()),
                                 porsi_tanda_hubung_host=float(g("has_hyphen_in_domain")[mask].mean()),
                                 median_panjang_url=float(np.median(g("url_length")[mask])),
                                 median_panjang_path=float(np.median(path[mask])),
                                 median_panjang_domain=float(np.median(g("domain_name_length")[mask]))))
            for k in np.unique(kategori):
                mk = legit & (kategori == k)
                cat.append(dict(split=r.kind, seed=r.seed, model=m, kategori=k, n_sah=int(mk.sum()),
                                porsi_sah=float(mk.sum() / legit.sum()),
                                **{f"fpr_{c}": float(fp[c][mk].mean()) if mk.any() else np.nan for c in CONDS},
                                **{f"n_fp_{c}": int(fp[c][mk].sum()) for c in CONDS}))
            if r.seed == 42 and m == "LightGBM":
                new = np.where(fp["adaptif"] & ~fp["tanpa"])[0]
                for i in new[:60]:
                    contoh.append(dict(split=r.kind, url=urls[i], kategori=kategori[i]))
    prof, cat = pd.DataFrame(prof), pd.DataFrame(cat)
    save(agg(prof, ["split", "model", "kelompok"]), "r2_4_profil_fp.csv")
    pooled = prof.groupby(["split", "kelompok"], sort=False).apply(
        lambda d: pd.Series({c: np.average(d[c], weights=d["n"]) for c in d.columns if c.startswith(("porsi", "median"))}
                            | {"n_total": d["n"].sum()}), include_groups=False).reset_index()
    save(pooled, "r2_4_profil_fp_gabungan.csv")
    save(agg(cat, ["split", "model", "kategori"]), "r2_4_fpr_per_kategori.csv")
    pc = cat.groupby(["split", "kategori"], sort=False)[[c for c in cat.columns if c.startswith("n_")]].sum().reset_index()
    for c in CONDS:
        pc[f"fpr_{c}"] = pc[f"n_fp_{c}"] / pc["n_sah"]
        pc[f"porsi_fp_{c}"] = pc[f"n_fp_{c}"] / pc.groupby("split")[f"n_fp_{c}"].transform("sum")
    pc["porsi_sah"] = pc["n_sah"] / pc.groupby("split")["n_sah"].transform("sum")
    save(pc, "r2_4_fpr_per_kategori_gabungan.csv")
    save(pd.DataFrame(contoh), "r2_4_contoh_fp_baru_lightgbm_seed42.csv")


def t_ext(runs):
    """R2-5: IDN homoglyph, open redirect and custom short link (extension)."""
    rows, pool = [], []
    for r in runs:
        if r.kind != "acak":
            continue
        z = r.z("utama_terkoreksi")
        ext = [str(s) for s in z["ext_names"]]
        for m in MODELS:
            for cond in CONDS + ["separuh"]:
                if f"terkoreksi__{m}__{cond}__ext" not in z.files:
                    continue
                P, E, det = r.cand("terkoreksi", m, cond), r.cand("terkoreksi", m, cond, "ext"), r.det("terkoreksi", m, cond)
                for j, name in enumerate(ext):
                    if " + " not in name:
                        rows.append(dict(seed=r.seed, model=m, kondisi=cond, teknik=name,
                                         evasion_buta=float(((E[:, j] < 0.5) & det).sum() / max(det.sum(), 1))))
                idn = np.array(["idn_homoglyph" in n for n in ext])
                pool.append(dict(seed=r.seed, model=m, kondisi=cond, pool_65=evasion(P, det),
                                 hanya_idn_13=evasion(E[:, idn], det),
                                 pool_65_plus_idn_78=evasion(np.hstack([P, E[:, idn]]), det),
                                 pool_65_plus_semua_80=evasion(np.hstack([P, E]), det)))
    save(agg(pd.DataFrame(rows), ["model", "kondisi", "teknik"]), "r2_5_teknik_baru_buta.csv")
    save(agg(pd.DataFrame(pool), ["model", "kondisi"]), "r2_5_teknik_baru_adaptif.csv")


def t_loao_shap(runs):
    for r in runs:
        if r.kind != "acak" or r.seed != 42:
            continue
        z = r.z("utama_terkoreksi")
        if "shap__tanpa" in z.files:
            sh = pd.DataFrame({c: z[f"shap__{c}"] for c in CONDS}, index=FEATURES_FIXED)
            sh.index.name = "fitur"
            save(sh.sort_values("tanpa", ascending=False).reset_index(), "tabel_shap_lightgbm_seed42.csv")
        if r.has("loao"):
            L = r.z("loao")
            base = dict(zip(r.attacks, blind_table(r, "terkoreksi", "LightGBM", "tanpa")))
            rows = []
            for held in BASE_ATTACKS:
                det = L[f"loao__{held}__clean"] >= 0.5
                rows.append(dict(teknik_disisihkan=held, evasion_tanpa_pertahanan=float(base[held]),
                                 evasion_loao=float(((L[f"loao__{held}__adv"] < 0.5) & det).sum() / max(det.sum(), 1))))
            save(pd.DataFrame(rows), "tabel10_leave_one_attack_out_seed42.csv")


def t_ngram_ringkas(bersih, buta, adapt):
    by = ["split", "set_fitur", "model", "kondisi"]
    b = bersih[bersih.set_fitur == "ngram"]
    if not len(b):
        return
    df = (b[by + ["seed", "f1", "recall", "fpr", "roc_auc"]]
          .merge(buta[buta.teknik == "RATA-RATA"][by + ["seed", "evasion"]].rename(columns={"evasion": "evasion_buta"}), on=by + ["seed"])
          .merge(adapt[by + ["seed", "evasion_adaptif"]], on=by + ["seed"]))
    save(agg(df, by), "r1_5_ngram_tfidf.csv")


def t_data(runs):
    r = next(x for x in runs if x.kind == "acak")
    m = {k: v for k, v in r.meta.items() if k != "contoh"}
    rows = [dict(keterangan=k, nilai=v) for k, v in m.items()]
    if not r.meta.get("n") or r.meta["n"] == r.meta["setelah_dedup_kanonik"]:
        from jalankan import load_data, safe_path
        df, _ = load_data()
        old = df.url.map(registered_domain)
        https = df.url.str.lower().str.startswith("https://")
        plen = df.url.map(lambda u: len(safe_path(u)))
        noq = ~df.url.str.contains("?", regex=False)
        sah, ph = df.y == 0, df.y == 1
        rows += [dict(keterangan="porsi_sah_beranda_https", nilai=float((https & (plen <= 1) & noq)[sah].mean())),
                 dict(keterangan="porsi_sah_ada_path", nilai=float((plen > 1)[sah].mean())),
                 dict(keterangan="porsi_phishing_ada_path", nilai=float((plen > 1)[ph].mean())),
                 dict(keterangan="porsi_phishing_https", nilai=float(https[ph].mean()))]
        rows += [dict(keterangan="url_domain_terdaftar_berubah_oleh_psl", nilai=int((old != df.domain).sum())),
                 dict(keterangan="domain_unik_daftar_manual", nilai=int(old.nunique())),
                 dict(keterangan="domain_unik_psl", nilai=int(df.domain.nunique())),
                 dict(keterangan="url_phishing", nilai=int(df.y.sum())),
                 dict(keterangan="url_sah", nilai=int((df.y == 0).sum()))]
    for x in runs:
        rows.append(dict(keterangan=f"{x.kind}_{x.seed}_latih_uji_phishing_uji_domain_uji",
                         nilai=f"{x.meta['n_latih']}/{x.meta['n_uji']}/{x.meta['uji_phishing']}/{x.meta['domain_uji']}"))
        rows.append(dict(keterangan=f"{x.kind}_{x.seed}_latih_phishing", nilai=x.meta["latih_phishing"]))
    save(pd.DataFrame(rows), "data_ringkasan.csv")
    with open(os.path.join(OUT, "contoh_url_samaran_seed42.json"), "w", encoding="utf-8") as f:
        json.dump(r.meta["contoh"], f, ensure_ascii=False, indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suf", default="")
    ap.add_argument("--seeds", default="42,7,2025")
    a = ap.parse_args()
    global OUT
    if a.suf:
        OUT = OUT + a.suf
    os.makedirs(OUT, exist_ok=True)
    runs = []
    for kind in ["acak", "domain"]:
        for s in map(int, a.seeds.split(",")):
            if os.path.exists(os.path.join(CACHE, f"{kind}_{s}{a.suf}__utama_terkoreksi.npz")):
                runs.append(Run(kind, s, a.suf))
    print("run tersedia:", [(r.kind, r.seed) for r in runs], "->", OUT)
    bersih, buta, adapt = t_utama(runs)
    t_ngram_ringkas(bersih, buta, adapt)
    t_heldout(runs)
    t_ambang(runs)
    t_logreg(runs)
    t_tiers(runs)
    t_budget(runs)
    t_ablasi(runs)
    t_fp(runs)
    t_ext(runs)
    t_loao_shap(runs)
    t_data(runs)


if __name__ == "__main__":
    main()
