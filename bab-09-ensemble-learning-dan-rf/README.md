# Bab 09 - Ensemble Learning dan Random Forest

> **Portofolio Tugas dan Praktikum Machine Learning Unindra**  
> Mahasiswa: **Anggit Arisa Siwi Pratama** | NPM: **`202343570045`** | Kelas: **XC Ekstensi**  
> Dosen: **Nurfidah Dwitiyanti, M.Si.** | Semester Gasal 2026/2027  

---

## Ringkasan Teknis

- **Topik Utama**: Bagging vs Boosting, Random Forest, AdaBoost, Gradient Boosting, Out-of-Bag (OOB) Score, Gini Feature Importance
- **Dataset**: `Telco Churn Customers (tenure, charges, support_calls, contract, internet_fiber)`
- **Jadwal Pelaksanaan**: 30 - 31 Oktober 2026

## Struktur Berkas

```text
bab-09-ensemble-learning-dan-rf/
├── main.py                                      # Skrip executable Python mandiri
├── bab_09_bab_09_ensemble_learning_dan_rf.ipynb # Jupyter Notebook terpadu & komprehensif
├── 09_ensemble_learning_dan_rf.ipynb            # Notebook modul utama Unindra
├── 09_alternatif_telco_churn_ensemble_methods.ipynb # Notebook studi kasus alternatif
├── LAPORAN_BAB_09_ANGGIT_ARISA_SIWI_PRATAMA_202343570045.pdf # Dokumen Laporan Resmi Cetak
├── laporan_printable.html                       # Template HTML cetak laporan
├── materi/                                      # Arsip materi resmi perkuliahan
│   └── .gitkeep
├── visual_output/                               # Grafik visual hasil komputasi
│   ├── 09_ensemble_methods_comparison.png
│   └── 09_feature_importance_rf.png
└── README.md                                    # Dokumentasi modul
```

## Panduan Menjalankan

```bash
# Menjalankan skrip simulasi mandiri
python main.py

# Membuka Jupyter Notebook
jupyter notebook bab_09_bab_09_ensemble_learning_dan_rf.ipynb
```

## Hasil Eksekusi Terminal

```text
[OK] Dataset Telco Churn dimuat: 600 pelanggan.

==================================================
HASIL EVALUASI ENSEMBLE LEARNING (BAB 09)
==================================================
1. Evaluasi Komparasi Model Ensemble:
- Decision Tree Baseline : Akurasi = 78.33% | F1 = 62.50%
- Random Forest Bagging   : Akurasi = 86.67% | F1 = 76.92% (OOB Score: 85.21%)
- AdaBoost                : Akurasi = 84.17% | F1 = 73.17%
- Gradient Boosting       : Akurasi = 85.83% | F1 = 75.61%

2. Fitur Paling Berpengaruh (Random Forest):
- tenure                 : 0.3421
- monthly_charges        : 0.2815
- total_charges          : 0.2110

[OK] Grafik visualisasi disimpan ke visual_output/09_ensemble_methods_comparison.png
[OK] Grafik visualisasi disimpan ke visual_output/09_feature_importance_rf.png
```

---
*Laboratorium Machine Learning - Teknik Informatika Universitas Indraprasta PGRI*
