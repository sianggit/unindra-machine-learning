# Bab 04 — Rekayasa Fitur Domain & ColumnTransformer Pipeline Bebas Leakage

> **Portofolio Tugas & Praktikum Machine Learning Unindra**  
> Mahasiswa: **Anggit Arisa Siwi Pratama** | NPM: **`202343570045`** | Kelas: **XC Ekstensi**  
> Dosen: **Nurfidah Dwitiyanti, M.Kom.** | Semester Gasal 2026/2027  

---

## 📌 Ringkasan Teknis

- **Topik Utama**: ColumnTransformer 3-Path, OneHotEncoder, OrdinalEncoder, Logistic Regression vs Random Forest
- **Dataset**: `Titanic Passenger Manifest N=891 (c/titanic)`
- **Jadwal Pelaksanaan**: 02 Oktober 2026

## 📁 Struktur Berkas

```
bab-04-feature-engineering-pipeline/
├── main.py                         # Skrip executable Python mandiri
├── bab_04_bab_04_feature_engineering_pipeline.ipynb  # Jupyter Notebook komprehensif
├── LAPORAN_BAB_04_ANGGIT_ARISA_SIWI_PRATAMA_202343570045.pdf # PDF Laporan Resmi
├── visual_output/
│   └── confusion_matrix_heatmap.png         # Grafik visual hasil pemodelan
└── README.md                       # Dokumentasi modul
```

## 🚀 Panduan Menjalankan

```bash
# Menjalankan skrip Python
python main.py

# Membuka Jupyter Notebook
jupyter notebook bab_04_bab_04_feature_engineering_pipeline.ipynb
```

## 📊 Hasil Eksekusi

```
[OK] Dataset Titanic dimuat dari CSV: 891 baris.
[Hasil Evaluasi Pipeline Bebas Data Leakage (Test N=179)]
1. Logistic Regression Akurasi : 76.54% | F1: 0.7407
2. Random Forest Akurasi       : 75.98% | F1: 0.7152
[OK] Grafik visual disimpan ke visual_output/confusion_matrix_heatmap.png
```

---
*Laboratorium Machine Learning — Teknik Informatika Universitas Indraprasta PGRI*
