# Bab 05 — Pohon Keputusan CART & C4.5 serta Cost-Complexity Pruning

> **Portofolio Tugas & Praktikum Machine Learning Unindra**  
> Mahasiswa: **Anggit Arisa Siwi Pratama** | NPM: **`202343570045`** | Kelas: **XC Ekstensi**  
> Dosen: **Nurfidah Dwitiyanti, M.Kom.** | Semester Gasal 2026/2027  

---

## 📌 Ringkasan Teknis

- **Topik Utama**: Gini Impurity, Information Gain, Cost-Complexity Alpha Pruning, Feature Importance
- **Dataset**: `UMKM Microloan Application Credit (mirichoi0218/insurance)`
- **Jadwal Pelaksanaan**: 09 Oktober 2026

## 📁 Struktur Berkas

```
bab-05-decision-tree/
├── main.py                         # Skrip executable Python mandiri
├── bab_05_bab_05_decision_tree.ipynb  # Jupyter Notebook komprehensif
├── LAPORAN_BAB_05_ANGGIT_ARISA_SIWI_PRATAMA_202343570045.pdf # PDF Laporan Resmi
├── visual_output/
│   └── feature_importance_chart.png         # Grafik visual hasil pemodelan
└── README.md                       # Dokumentasi modul
```

## 🚀 Panduan Menjalankan

```bash
# Menjalankan skrip Python
python main.py

# Membuka Jupyter Notebook
jupyter notebook bab_05_bab_05_decision_tree.ipynb
```

## 📊 Hasil Eksekusi

```
[OK] Dataset UMKM Microloan dimuat dari CSV: 450 baris.
Akurasi Pohon Keputusan Uji : 71.68%

Aturan Keputusan Pohon Terpangkas:
|--- lama_usaha_tahun <= 2.25
|   |--- class: 0
|--- lama_usaha_tahun >  2.25
|   |--- class: 1

[OK] Grafik visual disimpan ke visual_output/feature_importance_chart.png
```

---
*Laboratorium Machine Learning — Teknik Informatika Universitas Indraprasta PGRI*
