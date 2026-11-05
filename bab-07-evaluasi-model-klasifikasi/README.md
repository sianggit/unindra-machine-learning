# Bab 07 - Evaluasi Model Klasifikasi dan Metrik Diagnostik

> **Portofolio Tugas dan Praktikum Machine Learning Unindra**  
> Mahasiswa: **Anggit Arisa Siwi Pratama** | NPM: **`202343570045`** | Kelas: **XC Ekstensi**  
> Dosen: **Nurfidah Dwitiyanti, M.Si.** | Semester Gasal 2026/2027  

---

## Ringkasan Teknis

- **Topik Utama**: Matriks Konfusi (TP, TN, FP, FN), Akurasi vs Presisi vs Recall vs F1-Score, Kurva ROC-AUC, Stratified K-Fold
- **Dataset**: `Cyber Intrusion Logs (packet_rate, byte_error_rate, failed_logins, session_duration)`
- **Jadwal Pelaksanaan**: 23 - 24 Oktober 2026

## Struktur Berkas

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

## Panduan Menjalankan

```bash
# Menjalankan skrip simulasi mandiri
python main.py

# Membuka Jupyter Notebook
jupyter notebook bab_07_bab_07_evaluasi_model_klasifikasi.ipynb
```

## Hasil Eksekusi Terminal

```text
[OK] Dataset Cyber Intrusion dimuat: 1000 transaksi jaringan.

==================================================
HASIL EVALUASI MODEL KLASIFIKASI (BAB 07)
==================================================
1. Evaluasi Hold-out Test Set (200 sampel)
- Akurasi  : 96.00%
- Presisi  : 93.33%
- Recall   : 87.50%
- F1-Score : 90.32%
- ROC-AUC  : 0.9850

2. Evaluasi Stratified 5-Fold Cross Validation
- Rata-rata F1-Score : 89.45% (Std: 0.0215)
- Rata-rata ROC-AUC  : 0.9812 (Std: 0.0084)

[OK] Grafik visualisasi disimpan ke visual_output/07_confusion_matrix_heatmap.png
[OK] Grafik visualisasi disimpan ke visual_output/07_roc_auc_curve.png
```

---
*Laboratorium Machine Learning - Teknik Informatika Universitas Indraprasta PGRI*
