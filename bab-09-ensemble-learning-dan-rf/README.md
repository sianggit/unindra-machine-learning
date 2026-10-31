# Bab 09 — Ensemble Learning & Random Forest

> **Portofolio Tugas & Praktikum Machine Learning Unindra**  
> Mahasiswa: **Anggit Arisa Siwi Pratama** | NPM: **`202343570045`** | Kelas: **XC Ekstensi**  
> Dosen: **Nurfidah Dwitiyanti, M.Si.** | Semester Gasal 2026/2027  

---

## 📌 Ringkasan Teknis

- **Topik Utama**: Bagging vs Boosting, Random Forest, AdaBoost, Gradient Boosting, Out-of-Bag (OOB) Score, Gini Feature Importance
- **Dataset**: `Telco Churn Customers (tenure, charges, support_calls, contract, internet_fiber)`
- **Jadwal Pelaksanaan**: 30 – 31 Oktober 2026 (Pertemuan 9 - Daring LMS)

## 📁 Struktur Berkas

```text
bab-09-ensemble-learning-dan-rf/
├── main.py                                      # Skrip executable Python mandiri
├── bab_09_bab_09_ensemble_learning_dan_rf.ipynb # Jupyter Notebook terpadu & komprehensif
├── 09_ensemble_learning_dan_rf.ipynb            # Notebook modul utama Unindra
├── 09_alternatif_telco_churn_ensemble_methods.ipynb # Notebook studi kasus alternatif
├── LAPORAN_BAB_09_ANGGIT_ARISA_SIWI_PRATAMA_202343570045.pdf # Dokumen Laporan Resmi Cetak
├── laporan_printable.html                       # Template HTML cetak laporan
├── materi/                                      # Arsip materi & lembar modul perkuliahan
│   └── Modul_09_Ensemble_Learning_dan_RF.pdf
├── visual_output/                               # Grafik visual hasil komputasi
│   ├── 09_ensemble_methods_comparison.png
│   └── 09_feature_importance_rf.png
└── README.md                                    # Dokumentasi modul
```

## 🚀 Panduan Menjalankan

```bash
# Menjalankan skrip simulasi mandiri
python main.py

# Membuka Jupyter Notebook
jupyter notebook bab_09_bab_09_ensemble_learning_dan_rf.ipynb
```

## 📊 Hasil Eksekusi Terminal

```text
[OK] Dataset Telco Churn dimuat: 600 pelanggan.

==================================================
HASIL EVALUASI ENSEMBLE LEARNING (BAB 09)
==================================================
Single Tree (Baseline)   | Akurasi:  91.33% | F1:  85.71%
Random Forest (Bagging)  | Akurasi:  89.33% | F1:  82.61%
AdaBoost (Boosting)      | Akurasi:  88.67% | F1:  80.90%
Gradient Boosting        | Akurasi:  88.67% | F1:  81.72%
Out-of-Bag (OOB) Score RF: 89.56% (Estimasi error OOB: 10.44%)
==================================================
[OK] Visual output Bab 09 berhasil disimpan di visual_output/
```

## 💡 Kesimpulan Analitis

1. **Reduksi Varians Model**: Penggabungan 100 estimator pohon secara acak menekan varians tanpa mengorbankan stabilitas keputusan klasifikasi.
2. **Estimasi Out-of-Bag (OOB)**: Skor OOB sebesar 89.56% memverifikasi bahwa evaluasi internal bootstrap sampling sejalan dengan akurasi pengujian data tak-terlihat.
3. **Interpretabilitas Bisnis**: Analisis Gini Feature Importance menegaskan bahwa variabel masa berlangganan (*tenure*) dan jenis kontrak bulanan merupakan faktor paling berpengaruh terhadap retensi pelanggan.
