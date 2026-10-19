# Bab 02 — Partisi Data Bebas Bias, K-Fold Stratifikasi & Diagnostik Tradeoff

> **Portofolio Tugas & Praktikum Machine Learning Unindra**  
> Mahasiswa: **Anggit Arisa Siwi Pratama** | NPM: **`202343570045`** | Kelas: **XC Ekstensi**  
> Dosen: **Nurfidah Dwitiyanti, M.Kom.** | Semester Gasal 2026/2027  

---

## 📌 Ringkasan Teknis

- **Topik Utama**: Workflow CRISP-DM, Stratified K-Fold vs Random Holdout, Evaluasi Generalisasi
- **Dataset**: `IBM HR Analytics Employee Attrition (pavansubhasht/ibm-hr-analytics)`
- **Jadwal Pelaksanaan**: 12 September 2026

## 📁 Struktur Berkas

```
bab-02-workflow-ml/
├── main.py                         # Skrip executable Python mandiri
├── bab_02_bab_02_workflow_ml.ipynb  # Jupyter Notebook komprehensif
├── LAPORAN_BAB_02_ANGGIT_ARISA_SIWI_PRATAMA_202343570045.pdf # PDF Laporan Resmi
├── visual_output/
│   └── cv_stability_barchart.png         # Grafik visual hasil pemodelan
└── README.md                       # Dokumentasi modul
```

## 🚀 Panduan Menjalankan

```bash
# Menjalankan skrip Python
python main.py

# Membuka Jupyter Notebook
jupyter notebook bab_02_bab_02_workflow_ml.ipynb
```

## 📊 Hasil Eksekusi

```
[OK] Dataset IBM HR Attrition dimuat dari CSV: 400 baris.
Stratified 5-Fold Akurasi : [92.5, 92.5, 88.75, 91.25, 90.0]%
Rata-rata Stratified       : 91.00% (Std: 0.0146)
Rata-rata Random K-Fold   : 90.75% (Std: 0.0170)
[OK] Grafik visual disimpan ke visual_output/cv_stability_barchart.png
```

---
*Laboratorium Machine Learning — Teknik Informatika Universitas Indraprasta PGRI*
