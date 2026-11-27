"""Figures of the revised manuscript, drawn from the tables in results/revisi/.

    python gambar.py [--suf _n8000]

Output (figures/revisi/): gambar1_tahapan.png, gambar3_evasion_buta.png,
gambar4_tradeoff.png, gambar5_shap.png. Figure 2 (attacker flow) is unchanged.
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#8a8a85", "xtick.color": "#3d3d3a", "ytick.color": "#3d3d3a",
                     "axes.labelcolor": "#3d3d3a", "legend.frameon": False, "savefig.bbox": "tight"})
CMAP = LinearSegmentedColormap.from_list("biru", ["#f5f9fe", "#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"])
C3 = ["#2a78d6", "#eb6834", "#1baf7a"]
MODELS = ["LogReg", "RandomForest", "XGBoost", "LightGBM"]
MLAB = {"LogReg": "Logistic Regression", "RandomForest": "Random Forest", "XGBoost": "XGBoost", "LightGBM": "LightGBM"}
ATT = ["force_https", "root_landing", "typosquat", "homoglyph", "brand_subdomain", "at_trick", "strip_query",
       "mimic_legit_path", "shortener", "combo"]
CONDS = ["tanpa", "acak", "adaptif"]
CLAB = ["Tanpa adversarial training", "Adversarial training acak", "Adversarial training adaptif"]
koma = FuncFormatter(lambda v, _: f"{v:g}".replace(".", ","))


def k(v, nd=1):
    return f"{v:.{nd}f}".replace(".", ",")


def gambar1(out, n_bersih):
    boxes = {  # name: (x, y, w, h, title, body, style)
        1: (0.02, 0.70, 0.215, 0.23, "1. Dataset", "LegitPhish\n101.219 URL", "abu"),
        2: (0.268, 0.70, 0.215, 0.23, "2. Prapemrosesan", f"kanonikalisasi URL,\nhapus duplikat\n→ {n_bersih} URL", "abu"),
        3: (0.516, 0.70, 0.215, 0.23, "3. Ekstraksi fitur", "set replikasi (16)\nset terkoreksi (18)", "biru"),
        4: (0.764, 0.70, 0.215, 0.23, "4. Pelatihan model", "LR, RF, XGBoost,\nLightGBM\nsplit 80:20, 3 seed", "biru"),
        5: (0.764, 0.36, 0.215, 0.23, "5. Evaluasi baseline", "data bersih\n(RQ1)", "biru"),
        6: (0.516, 0.36, 0.215, 0.23, "6. Serangan evasion", "buta: 10 teknik\nadaptif: 65 rantai\n(RQ2)", "oranye"),
        7: (0.268, 0.36, 0.215, 0.23, "7. Adversarial training", "acak & adaptif\n(RQ3)", "hijau"),
        8: (0.02, 0.36, 0.215, 0.23, "8. Analisis tambahan", "SHAP, split per domain,\nablasi, uji sensitivitas\n(RQ4)", "biru"),
        9: (0.03, 0.05, 0.43, 0.17, "9. Perbandingan set fitur", "replikasi vs terkoreksi (RQ5)", "biru"),
        10: (0.57, 0.05, 0.34, 0.17, "10. Pembahasan & kesimpulan", "Bagian 3.8 dan 4", "abu"),
    }
    style = {"abu": ("#f2f2f2", "#888888"), "biru": ("#eaf2fb", "#2a78d6"),
             "oranye": ("#fdeee6", "#eb6834"), "hijau": ("#e2f5ec", "#1baf7a")}
    fig, ax = plt.subplots(figsize=(10.4, 5.6))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    for x, y, w, h, title, body, st in boxes.values():
        fc, ec = style[st]
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.012", fc=fc, ec=ec, lw=2))
        nl = body.count("\n")
        ax.text(x + w / 2, y + h * (0.74 if nl else 0.66), title, ha="center", va="center", fontsize=10.5,
                fontweight="bold", color="#333333")
        ax.text(x + w / 2, y + h * (0.33 if nl else 0.32), body, ha="center", va="center", fontsize=9.5,
                color="#333333", linespacing=1.2)

    def arrow(a, b):
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=16, lw=2.2, color="#666666",
                                     shrinkA=0, shrinkB=0))
    B = boxes
    for i in (1, 2, 3):
        arrow((B[i][0] + B[i][2], B[i][1] + B[i][3] / 2), (B[i + 1][0], B[i + 1][1] + B[i + 1][3] / 2))
    arrow((B[4][0] + B[4][2] / 2, B[4][1]), (B[5][0] + B[5][2] / 2, B[5][1] + B[5][3]))
    for i in (5, 6, 7):
        arrow((B[i][0], B[i][1] + B[i][3] / 2), (B[i + 1][0] + B[i + 1][2], B[i + 1][1] + B[i + 1][3] / 2))
    arrow((B[8][0] + B[8][2] / 2, B[8][1]), (B[8][0] + B[8][2] / 2, B[9][1] + B[9][3]))
    arrow((B[9][0] + B[9][2], B[9][1] + B[9][3] / 2), (B[10][0], B[10][1] + B[10][3] / 2))
    fig.savefig(f"{out}/gambar1_tahapan.png", dpi=300, pad_inches=0.05); plt.close(fig)


def gambar3(res, out):
    t = pd.read_csv(f"{res}/tabel6_evasion_buta.csv")
    t = t[(t.split == "acak") & (t.set_fitur == "terkoreksi") & (t.kondisi == "tanpa")]
    tbl = t.pivot(index="teknik", columns="model", values="evasion_mean").loc[ATT + ["RATA-RATA"], MODELS] * 100
    tbl = tbl.rename(index={"RATA-RATA": "Rata-rata"})
    fig, ax = plt.subplots(figsize=(6.3, 4.6))
    im = ax.imshow(tbl.values, cmap=CMAP, vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(range(4), [MLAB[m].replace(" ", "\n") for m in MODELS]); ax.set_yticks(range(len(tbl)), tbl.index)
    ax.set_xticks(np.arange(-.5, 4), minor=True); ax.set_yticks(np.arange(-.5, len(tbl)), minor=True)
    ax.grid(which="minor", color="white", linewidth=2); ax.tick_params(which="both", length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    for i in range(tbl.shape[0]):
        for j in range(4):
            v = tbl.values[i, j]
            ax.text(j, i, k(v), ha="center", va="center", color="white" if v > 50 else "#1f1f1e", fontsize=8.5)
    cb = fig.colorbar(im, ax=ax, fraction=0.04); cb.set_label("Tingkat evasion (%)"); cb.outline.set_visible(False)
    fig.tight_layout(); fig.savefig(f"{out}/gambar3_evasion_buta.png", dpi=300); plt.close(fig)


def gambar4(res, out):
    a = pd.read_csv(f"{res}/tabel7_evasion_adaptif.csv")
    b = pd.read_csv(f"{res}/tabel9_bersih_tiga_kondisi.csv")
    sel = lambda d: d[(d.split == "acak") & (d.set_fitur == "terkoreksi")].set_index(["model", "kondisi"])  # noqa: E731
    a, b = sel(a), sel(b)
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    marks = ["o", "s", "^"]
    xmax = 0
    for ci, m in enumerate(MODELS[1:]):
        x = [b.loc[(m, c), "fpr_mean"] * 100 for c in CONDS]
        y = [a.loc[(m, c), "evasion_adaptif_mean"] * 100 for c in CONDS]
        xmax = max(xmax, max(x))
        ax.plot(x, y, color=C3[ci], lw=1.8, alpha=0.85, zorder=2)
        for xi, yi, mk in zip(x, y, marks):
            ax.scatter([xi], [yi], marker=mk, s=60, color=C3[ci], edgecolor="white", linewidth=1, zorder=3)
            ax.annotate(k(yi), (xi, yi), textcoords="offset points", xytext=(8, 5), fontsize=7.5, color="#3d3d3a")
    ax.set_xlabel("False positive rate pada data uji bersih (%)"); ax.set_ylabel("Tingkat evasion adaptif (%)")
    ax.set_xlim(0, max(1.0, np.ceil(xmax * 10 + 1) / 10)); ax.set_ylim(0, None)
    ax.xaxis.set_major_formatter(koma); ax.yaxis.set_major_formatter(koma)
    ax.grid(True, color="#e4e3df", linewidth=0.8); ax.set_axisbelow(True)
    h1 = [Line2D([], [], color=C3[i], lw=2, label=MLAB[m]) for i, m in enumerate(MODELS[1:])]
    h2 = [Line2D([], [], color="#777777", marker=mk, lw=0, markersize=7, label=l) for mk, l in zip(marks, CLAB)]
    l1 = ax.legend(handles=h1, title="Model", loc="upper right", fontsize=8, title_fontsize=8)
    ax.add_artist(l1)
    ax.legend(handles=h2, title="Kondisi pelatihan", loc="center right", fontsize=8, title_fontsize=8)
    fig.tight_layout(); fig.savefig(f"{out}/gambar4_tradeoff.png", dpi=300); plt.close(fig)


def gambar5(res, out):
    t = pd.read_csv(f"{res}/tabel_shap_lightgbm_seed42.csv").set_index("fitur").sort_values("tanpa")
    fig, ax = plt.subplots(figsize=(6.3, 5.6)); y = np.arange(len(t))
    for i, (c, lab) in enumerate(zip(CONDS, CLAB)):
        ax.barh(y + (1 - i) * 0.27, t[c], height=0.25, color=C3[i], label=lab)
    ax.set_yticks(y, t.index); ax.set_xlabel("Rata-rata |nilai SHAP|"); ax.legend(loc="lower right", fontsize=8)
    ax.xaxis.set_major_formatter(koma)
    ax.xaxis.grid(True, color="#e4e3df", linewidth=0.8); ax.set_axisbelow(True); ax.tick_params(axis="y", length=0)
    fig.tight_layout(); fig.savefig(f"{out}/gambar5_shap.png", dpi=300); plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--suf", default="")
    a = ap.parse_args()
    res = os.path.join(ROOT, "results", "revisi" + a.suf)
    out = os.path.join(ROOT, "figures", "revisi" + a.suf)
    os.makedirs(out, exist_ok=True)
    d = pd.read_csv(f"{res}/data_ringkasan.csv").set_index("keterangan")["nilai"]
    gambar1(out, f"{int(d['setelah_dedup_kanonik']):,}".replace(",", "."))
    gambar3(res, out)
    gambar4(res, out)
    gambar5(res, out)
    print("gambar selesai:", out)
