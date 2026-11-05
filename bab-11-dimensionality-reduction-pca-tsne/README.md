# Bab 11 - Dimensionality Reduction: PCA dan t-SNE

> **Portofolio Tugas dan Praktikum Machine Learning Unindra**  
> Mahasiswa: **Anggit Arisa Siwi Pratama** | NPM: **`202343570045`** | Kelas: **XC Ekstensi**  
> Dosen: **Nurfidah Dwitiyanti, M.Si.** | Semester Gasal 2026/2027  

---

## Ringkasan Teknis

- **Topik Utama**: Mitigasi *Curse of Dimensionality*, Dekomposisi Spektral Matriks Kovariansi, Nilai Eigen dan Vektor Eigen, Principal Component Analysis (PCA), Rasio Variansi Terjelaskan (*Explained Variance Ratio*), dan Visualisasi Manifold Non-linear t-SNE.
- **Dataset**: `Scikit-Learn Optical Recognition of Handwritten Digits (N=1.797 sampel, 64 fitur) dan Olivetti Faces (4.096 dimensi)`
- **Jadwal Pelaksanaan**: 14 November 2026

## Struktur Berkas

```text
bab-11-dimensionality-reduction-pca-tsne/
├── main.py                                      # Skrip executable Python mandiri
├── 11_dimensionality_reduction_pca_tsne.ipynb   # Notebook modul utama Unindra
├── 11_alternatif_olivetti_faces_pca_tsne.ipynb  # Notebook studi kasus alternatif
├── LAPORAN_BAB_11_ANGGIT_ARISA_SIWI_PRATAMA_202343570045.pdf # Dokumen Laporan Resmi Cetak A4
├── visual_output/                               # Grafik visual hasil komputasi
│   └── 11_pca_2d_projection.png
└── README.md                                    # Dokumentasi modul
```

## Panduan Menjalankan

```bash
# Menjalankan skrip simulasi mandiri
python main.py

# Membuka Jupyter Notebook
jupyter notebook 11_dimensionality_reduction_pca_tsne.ipynb
```

## Hasil Eksekusi Terminal

```text
[OK] Dataset Digits dimuat: 1797 sampel, 64 dimensi fitur.

=======================================================
HASIL EVALUASI REDUKSI DIMENSI PCA & t-SNE (BAB 11)
=======================================================
Pembuktian Eigen Bab 11 (Hal 66): Lambda 1 = 1.0000, Lambda 2 = 0.5000
Explained Variance Ratio PC1    : 14.89%
Explained Variance Ratio PC2    : 13.62%
Total Variansi Terjelaskan (2D) : 28.51%
Reduksi Dimensi Fitur           : 64D -> 2D (Kompresi 96.88%)
=======================================================

[OK] Grafik visualisasi disimpan ke visual_output/11_pca_2d_projection.png
```

---
*Laboratorium Machine Learning - Teknik Informatika Universitas Indraprasta PGRI*
