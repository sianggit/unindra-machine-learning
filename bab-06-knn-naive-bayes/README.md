# Bab 06 — Klasifikasi Non-Parametrik K-NN & Bayesian Generative Model

> **Portofolio Tugas & Praktikum Machine Learning Unindra**  
> Mahasiswa: **Anggit Arisa Siwi Pratama** | NPM: **`202343570045`** | Kelas: **XC Ekstensi**  
> Dosen: **Nurfidah Dwitiyanti, M.Kom.** | Semester Gasal 2026/2027  

---

## 📌 Ringkasan Teknis

- **Topik Utama**: Distance-Weighted KNN (Manhattan vs Euclidean), Multinomial Naive Bayes Laplace Smoothing
- **Dataset**: `E-Commerce Reviews Sentiment (nicapotato/womens-ecommerce)`
- **Jadwal Pelaksanaan**: 16 Oktober 2026

## 📁 Struktur Berkas

```
bab-06-knn-naive-bayes/
├── main.py                         # Skrip executable Python mandiri
├── bab_06_bab_06_knn_naive_bayes.ipynb  # Jupyter Notebook komprehensif
├── LAPORAN_BAB_06_ANGGIT_ARISA_SIWI_PRATAMA_202343570045.pdf # PDF Laporan Resmi
├── visual_output/
│   └── model_benchmark_chart.png         # Grafik visual hasil pemodelan
└── README.md                       # Dokumentasi modul
```

## 🚀 Panduan Menjalankan

```bash
# Menjalankan skrip Python
python main.py

# Membuka Jupyter Notebook
jupyter notebook bab_06_bab_06_knn_naive_bayes.ipynb
```

## 📊 Hasil Eksekusi

```
[OK] Dataset E-Commerce Sentiment dimuat dari CSV: 300 ulasan.

[Hasil Evaluasi Komparatif Sentimen]
1. Multinomial Naive Bayes : 100.00% Akurasi | F1: 1.0000
2. Distance-Weighted K-NN  : 100.00% Akurasi | F1: 1.0000
[OK] Grafik visual disimpan ke visual_output/model_benchmark_chart.png
```

---
*Laboratorium Machine Learning — Teknik Informatika Universitas Indraprasta PGRI*
