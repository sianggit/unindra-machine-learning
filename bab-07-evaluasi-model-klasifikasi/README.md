# Bab 07 — Evaluasi Model Klasifikasi & Metrik Diagnostik

> **Portofolio Tugas & Praktikum Machine Learning Unindra**  
> Mahasiswa: **Anggit Arisa Siwi Pratama** | NPM: **`202343570045`** | Kelas: **XC Ekstensi**  
> Dosen: **Nurfidah Dwitiyanti, M.Si.** | Semester Gasal 2026/2027  

---

## 📌 Ringkasan Teknis

- **Topik Utama**: Matriks Konfusi (TP, TN, FP, FN), Akurasi vs Presisi vs Recall vs F1-Score, Kurva ROC-AUC, Stratified K-Fold
- **Dataset**: `Cyber Intrusion Logs (packet_rate, byte_error_rate, failed_logins, session_duration)`
- **Jadwal Pelaksanaan**: 23 – 24 Oktober 2026 (Pertemuan 7 - Luring, R.7.2-6)

## 📁 Struktur Berkas

```text
bab-07-evaluasi-model-klasifikasi/
├── main.py                                      # Skrip executable Python mandiri
├── bab_07_bab_07_evaluasi_model_klasifikasi.ipynb # Jupyter Notebook terpadu & komprehensif
├── 07_evaluasi_model_klasifikasi.ipynb          # Notebook modul utama Unindra
├── 07_alternatif_cyber_intrusion_evaluation_metrics.ipynb # Notebook studi kasus alternatif
├── LAPORAN_BAB_07_ANGGIT_ARISA_SIWI_PRATAMA_202343570045.pdf # Dokumen Laporan Resmi Cetak
├── laporan_printable.html                       # Template HTML cetak laporan
├── materi/                                      # Arsip materi & lembar modul perkuliahan
│   └── Modul_07_Evaluasi_Model_Klasifikasi.pdf
├── visual_output/                               # Grafik visual hasil komputasi
│   ├── 07_confusion_matrix_heatmap.png
│   └── 07_roc_auc_curve.png
└── README.md                                    # Dokumentasi modul
```

## 🚀 Panduan Menjalankan

```bash
# Menjalankan skrip simulasi mandiri
python main.py

# Membuka Jupyter Notebook
jupyter notebook bab_07_bab_07_evaluasi_model_klasifikasi.ipynb
```

## 📊 Hasil Eksekusi Terminal

```text
[OK] Dataset Cyber Intrusion dimuat: 1000 transaksi jaringan.

==================================================
HASIL EVALUASI MODEL KLASIFIKASI (BAB 07)
==================================================
True Positive (TP)  :   32  | False Positive (FP) :    0
False Negative (FN) :    0  | True Negative (TN)  :  218
Akurasi Uji         : 100.00%
Presisi (Precision) : 100.00%
Recall (Sensitivitas): 100.00%
F1-Score Harmonik   : 100.00%
ROC-AUC Score       : 1.0000
Stratified 5-Fold F1: 98.84% (+/- 0.95%)
==================================================
[OK] Visual output Bab 07 berhasil disimpan di visual_output/
```

## 💡 Kesimpulan Analitis

1. **Dilema Paradoks Akurasi**: Pada data tidak seimbang, akurasi tinggi dapat menipu. Presisi dan Recall membedah performa deteksi intrusi siber secara proporsional.
2. **Prioritas Keamanan Siber**: Memaksimalkan Recall (Sensitivitas) sangat vital guna menekan angka False Negative (serangan penyusup yang lolos).
3. **Stabilitas Stratified K-Fold**: Skor 5-fold F1 sebesar 98.84% menunjukkan kapabilitas generalisasi model yang kokoh dan konsisten.
