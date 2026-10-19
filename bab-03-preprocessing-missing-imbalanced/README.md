# Bab 03 — Imputasi Robust Nilai Rumpang & Penyeimbangan Kelas SMOTE

> **Portofolio Tugas & Praktikum Machine Learning Unindra**  
> Mahasiswa: **Anggit Arisa Siwi Pratama** | NPM: **`202343570045`** | Kelas: **XC Ekstensi**  
> Dosen: **Nurfidah Dwitiyanti, M.Kom.** | Semester Gasal 2026/2027  

---

## 📌 Ringkasan Teknis

- **Topik Utama**: Penanganan Missing Values Median, IQR Outlier Capping, Resampling SMOTE 1:1
- **Dataset**: `Financial Fraud Detection Transactions (ealaxi/paysim1)`
- **Jadwal Pelaksanaan**: 19 September 2026

## 📁 Struktur Berkas

```
bab-03-preprocessing-missing-imbalanced/
├── main.py                         # Skrip executable Python mandiri
├── bab_03_bab_03_preprocessing_missing_imbalanced.ipynb  # Jupyter Notebook komprehensif
├── LAPORAN_BAB_03_ANGGIT_ARISA_SIWI_PRATAMA_202343570045.pdf # PDF Laporan Resmi
├── visual_output/
│   └── smote_class_distribution.png         # Grafik visual hasil pemodelan
└── README.md                       # Dokumentasi modul
```

## 🚀 Panduan Menjalankan

```bash
# Menjalankan skrip Python
python main.py

# Membuka Jupyter Notebook
jupyter notebook bab_03_bab_03_preprocessing_missing_imbalanced.ipynb
```

## 📊 Hasil Eksekusi

```
[OK] Dataset Financial Fraud dimuat dari CSV: 600 baris.
[Preprocessing & Penyeimbangan Kelas Selesai]
- Missing Value Pasca-Imputasi       : 0
- Distribusi Kelas Sebelum Resampling: Normal=581, Fraud=19
- Distribusi Kelas Pasca-Resampling  : Normal=581, Fraud=581
[OK] Grafik visual disimpan ke visual_output/smote_class_distribution.png
```

---
*Laboratorium Machine Learning — Teknik Informatika Universitas Indraprasta PGRI*
