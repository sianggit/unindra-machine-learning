# Bab 01 — Pengantar Machine Learning, Lingkungan Komputasi & Matriks Kovariansi

> **Portofolio Tugas & Praktikum Machine Learning Unindra**  
> Mahasiswa: **Anggit Arisa Siwi Pratama** | NPM: **`202343570045`** | Kelas: **XC Ekstensi**  
> Dosen: **Nurfidah Dwitiyanti, M.Kom.** | Semester Gasal 2026/2027  

---

## 📌 Ringkasan Teknis

- **Topik Utama**: NumPy Vectorization, Z-Score Standardizer, Matriks Kovariansi Morfometrik
- **Dataset**: `Palmer Archipelago Penguins (parulpandey/palmer-archipelago)`
- **Jadwal Pelaksanaan**: 05 September 2026

## 📁 Struktur Berkas

```
bab-01-pengantar-ml/
├── main.py                         # Skrip executable Python mandiri
├── bab_01_bab_01_pengantar_ml.ipynb  # Jupyter Notebook komprehensif
├── LAPORAN_BAB_01_ANGGIT_ARISA_SIWI_PRATAMA_202343570045.pdf # PDF Laporan Resmi
├── visual_output/
│   └── covariance_matrix_heatmap.png         # Grafik visual hasil pemodelan
└── README.md                       # Dokumentasi modul
```

## 🚀 Panduan Menjalankan

```bash
# Menjalankan skrip Python
python main.py

# Membuka Jupyter Notebook
jupyter notebook bab_01_bab_01_pengantar_ml.ipynb
```

## 📊 Hasil Eksekusi

```
[OK] Dataset Palmer Penguins dimuat dari CSV: 300 baris, 4 fitur.

Matriks Kovariansi Fitur Morfometrik Terstandarisasi:
[[ 1.0033 -0.3495  0.4868  0.4114]
 [-0.3495  1.0033 -0.7132 -0.6894]
 [ 0.4868 -0.7132  1.0033  0.7208]
 [ 0.4114 -0.6894  0.7208  1.0033]]

[Selesai] Dimensi Matriks Fitur: 300 baris x 4 kolom.
[OK] Grafik visual disimpan ke visual_output/covariance_matrix_heatmap.png
```

---
*Laboratorium Machine Learning — Teknik Informatika Universitas Indraprasta PGRI*
