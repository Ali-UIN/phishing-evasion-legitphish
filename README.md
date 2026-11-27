# Phishing URL Evasion — LegitPhish

Judul: **Evaluating and Improving Phishing URL Detector Robustness Against Evasion Attacks on LegitPhish**

Celah penelitian dari Potpelwar dkk. (2025), *Data in Brief* 63, 111972 (doi:10.1016/j.dib.2025.111972):
dataset belum memuat URL phishing yang sengaja disamarkan, dan uji ketahanan terhadap evasion disarankan sebagai future work.

## Isi folder
| File | Fungsi |
|---|---|
| `eksperimen.ipynb` | rancangan (RQ1–RQ5) + seluruh eksperimen, jalankan dari atas ke bawah |
| `features.py` | dua set fitur dari URL mentah: **replikasi** (16 fitur, rumus asli dataset, cocok ≥99,97%) dan **terkoreksi** (parsing diperbaiki + `has_port`, `no_scheme`) |
| `perturb.py` | 10 teknik penyamaran URL (force_https, root_landing, typosquat, homoglyph, brand_subdomain, at_trick, strip_query, mimic_legit_path, shortener, combo) + generator kandidat penyerang adaptif (65 rantai) |
| `data/` | CSV LegitPhish |
| `results/<set>/`, `figures/<set>/` | dibuat otomatis per set fitur; `results/tabel8_replikasi_vs_terkoreksi.csv` membandingkan keduanya |

## Cara jalan
1. CSV LegitPhish (Mendeley Data, https://data.mendeley.com/datasets/hx4m73v2sf/2, CC BY 4.0) taruh di `data/`.
2. Pakai venv `owlert\ml\.venv` (sudah punya semua paket), atau buat baru dan `pip install -r requirements.txt`.
3. Buka `eksperimen.ipynb` di VS Code, pilih kernel venv, **Run All** dengan `FEATURE_SET = "replikasi"`.
4. Ganti jadi `FEATURE_SET = "terkoreksi"` di sel Setup, **Run All** lagi. Sel terakhir membuat tabel perbandingan.
   Perkiraan 15–20 menit per set di data penuh (bagian penyerang adaptif paling lama).

## Catatan
- Kelas positif = phishing (label asli dataset 0 = phishing, dibalik di notebook).
- URL duplikat dan URL berlabel ganda dibuang sebelum split.
- Serangan hanya diterapkan pada URL phishing di test set; path legit untuk `mimic_legit_path` diambil dari data latih saja.
- Penyerang adaptif: black-box berbasis kueri, mencoba 65 rantai penyamaran per URL dan memilih yang skor phishing-nya terendah.
- Sudah dites end-to-end pada subsampel 12.000 URL asli untuk kedua set fitur; angka final harus dari run penuh.

## Revisi (respons reviewer, November 2026)
Folder `revisi/` berisi eksperimen tambahan untuk revisi naskah: kanonikalisasi URL (RFC 3986) sebelum deduplikasi,
domain terdaftar dengan Mozilla Public Suffix List, tiga seed (42, 7, 2025) pada split acak dan split per domain,
keluarga rantai yang disisihkan dari pelatihan, penyetelan ambang dan kalibrasi, anggaran kueri dan penyerang hard-label,
tingkat kelayakan teknik, ablasi fitur, analisis galat false positive, teknik IDN homoglyph / open redirect / custom
shortlink, pembanding n-gram karakter, dan uji lintas-dataset pada PhiUSIIL.
Notebook dan modul di atas tidak diubah. Cara menjalankan ada di `revisi/README.md`; tabel hasil di `results/revisi/`,
gambar di `figures/revisi/`.
