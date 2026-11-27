# Eksperimen revisi (pipeline v2)

Eksperimen tambahan untuk menjawab komentar dua reviewer Jurnal Informatika Press.
Notebook `eksperimen.ipynb`, `features.py`, dan `perturb.py` di folder induk tidak diubah.

## Yang berbeda dari pipeline awal
- URL dikanonikalisasi menurut RFC 3986 sebelum deduplikasi (100.866 URL).
- Domain terdaftar dan public suffix memakai Mozilla Public Suffix List, bagian ICANN
  (`psl_snapshot.txt`, commit `1cbd6e71a9b83620b1d0b11e49d3d9ff48c27e22`, lisensi MPL 2.0).
- Semua eksperimen serangan dan pertahanan diulang pada seed 42, 7, 2025, pada split acak
  dan split per domain.

## Berkas
| Berkas | Fungsi |
|---|---|
| `psl.py`, `psl_snapshot.txt`, `buat_psl_snapshot.py` | parser Public Suffix List dan salinan luringnya |
| `features_v2.py` | kanonikalisasi URL, fitur terkoreksi dengan PSL, tiga fitur host |
| `perturb_v2.py` | teknik penyamaran dengan PSL, tiga teknik perluasan (IDN homoglyph, open redirect, custom shortlink) |
| `jalankan.py` | melatih model dan menyimpan skor mentah ke `cache/` (bagian berat) |
| `analisis.py` | mengubah skor menjadi tabel di `../results/revisi/` |
| `waktu_prediksi.py` | waktu prediksi per URL untuk Tabel 5 |
| `lintas_dataset.py` | uji lintas-dataset: model dilatih di LegitPhish, diuji di PhiUSIIL tanpa pelatihan ulang |
| `gambar.py` | Gambar 1, 3, 4, 5 ke `../figures/revisi/` |

## Cara menjalankan
Pakai lingkungan Python yang sama dengan notebook (pandas, numpy, scikit-learn, lightgbm, xgboost, shap, matplotlib, scipy).

```
python jalankan.py --kind acak --seed 42      # ulangi untuk seed 7 dan 2025
python jalankan.py --kind domain --seed 42    # ulangi untuk seed 7 dan 2025
python analisis.py
python waktu_prediksi.py
python lintas_dataset.py
python gambar.py
```

`lintas_dataset.py` butuh CSV PhiUSIIL (UCI Machine Learning Repository, dataset 967, CC BY 4.0) di
`../data/eksternal/PhiUSIIL_Phishing_URL_Dataset.csv`. Berkas itu tidak ikut di repositori.

Satu run `acak` memakan sekitar 30 menit dan satu run `domain` sekitar 10 menit pada laptop 6 inti.
Folder `cache/` (skor mentah, sekitar 1 GB) tidak perlu disimpan di repositori; semua tabel dapat
dibangun ulang darinya. Uji cepat: `python jalankan.py --seed 42 --n 8000`, lalu
`python analisis.py --suf _n8000 --seeds 42`.

## Peta komentar reviewer ke tabel hasil
| Komentar | Tabel di `results/revisi/` |
|---|---|
| R1-1 kanonikalisasi, tiga seed | `data_ringkasan.csv`, `tabel5_baseline.csv`, `tabel7_evasion_adaptif.csv`, `tabel9_bersih_tiga_kondisi.csv` |
| R1-2 Public Suffix List | `data_ringkasan.csv` |
| R1-3 rantai yang disisihkan, ambang | `r1_3_rantai_heldout.csv`, `r1_3_ambang.csv` |
| R1-4 koefisien Logistic Regression | `r1_4_koefisien_logreg.csv`, `r1_4_campuran_rantai.csv`, `r1_4_logreg_lolos_per_rantai.csv` |
| R1-5 split per domain, model n-gram karakter | baris `split = domain` pada tabel 7 dan 9, `r1_5_ngram_tfidf.csv` |
| R2-1 tingkat kelayakan | `r2_1_tingkat_kelayakan.csv` |
| R2-2 anggaran kueri, hard-label | `r2_2_anggaran_kueri.csv`, `r2_2_hard_label.csv` |
| R2-3 ablasi fitur | `r2_3_ablasi.csv` |
| R2-4 analisis galat false positive, uji lintas-dataset | `r2_4_*.csv` (lintas-dataset: `r2_4_lintas_dataset*.csv`) |
| R2-5 teknik perluasan | `r2_5_teknik_baru_buta.csv`, `r2_5_teknik_baru_adaptif.csv` |
