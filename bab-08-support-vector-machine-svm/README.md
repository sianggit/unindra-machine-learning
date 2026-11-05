# Bab 08 - Support Vector Machine (SVM)

> **Portofolio Tugas dan Praktikum Machine Learning Unindra**  
> Mahasiswa: **Anggit Arisa Siwi Pratama** | NPM: **`202343570045`** | Kelas: **XC Ekstensi**  
> Dosen: **Nurfidah Dwitiyanti, M.Si.** | Semester Gasal 2026/2027  

---

## Ringkasan Teknis

- **Topik Utama**: Maximal Margin Hyperplane ($w^T x + b = 0$), Lebar Margin Geometris ($M = 2/\|w\|$), Support Vectors, Komparasi Kernel (Linear, RBF, Poly)
- **Dataset**: `Solar Panel Faults (irradiance, temperature_c, voltage_v, current_a, is_fault)`
- **Jadwal Pelaksanaan**: 25 Oktober 2026

## Struktur Berkas

```text
bab-08-support-vector-machine-svm/
├── main.py                                      # Skrip executable Python mandiri
├── bab_08_bab_08_support_vector_machine_svm.ipynb # Jupyter Notebook terpadu & komprehensif
├── 08_support_vector_machine_svm.ipynb          # Notebook modul utama Unindra
├── 08_alternatif_solar_panel_fault_svm_kernels.ipynb # Notebook studi kasus alternatif
├── LAPORAN_BAB_08_ANGGIT_ARISA_SIWI_PRATAMA_202343570045.pdf # Dokumen Laporan Resmi Cetak
├── laporan_printable.html                       # Template HTML cetak laporan
├── materi/                                      # Arsip materi resmi perkuliahan
│   └── .gitkeep
├── visual_output/                               # Grafik visual hasil komputasi
│   ├── 08_svm_hyperplane_margins.png
│   └── 08_svm_kernel_comparison.png
└── README.md                                    # Dokumentasi modul
```

## Panduan Menjalankan

```bash
# Menjalankan skrip simulasi mandiri
python main.py

# Membuka Jupyter Notebook
jupyter notebook bab_08_bab_08_support_vector_machine_svm.ipynb
```

## Hasil Eksekusi Terminal

```text
[OK] Dataset Solar Panel dimuat: 400 titik pantau modul surya.

==================================================
HASIL EVALUASI SUPPORT VECTOR MACHINE (BAB 08)
==================================================
1. Evaluasi Komparasi Kernel SVM:
- Linear Kernel : Akurasi = 87.50% | F1 = 86.21%
- RBF Kernel    : Akurasi = 95.00% | F1 = 94.74% (Optimal)
- Poly Kernel   : Akurasi = 91.25% | F1 = 90.67%

2. Parameter Model Terbaik (RBF Kernel):
- Jumlah Support Vectors: 78
- Lebar Margin Efektif  : 0.4521

[OK] Grafik visualisasi disimpan ke visual_output/08_svm_hyperplane_margins.png
[OK] Grafik visualisasi disimpan ke visual_output/08_svm_kernel_comparison.png
```

---
*Laboratorium Machine Learning - Teknik Informatika Universitas Indraprasta PGRI*
