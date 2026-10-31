# Bab 08 — Support Vector Machine (SVM)

> **Portofolio Tugas & Praktikum Machine Learning Unindra**  
> Mahasiswa: **Anggit Arisa Siwi Pratama** | NPM: **`202343570045`** | Kelas: **XC Ekstensi**  
> Dosen: **Nurfidah Dwitiyanti, M.Si.** | Semester Gasal 2026/2027  

---

## 📌 Ringkasan Teknis

- **Topik Utama**: Maximal Margin Hyperplane ($w^T x + b = 0$), Lebar Margin Geometris ($M = 2/\|w\|$), Support Vectors, Komparasi Kernel (Linear, RBF, Poly)
- **Dataset**: `Solar Panel Faults (irradiance, temperature_c, voltage_v, current_a, is_fault)`
- **Jadwal Pelaksanaan**: 25 Oktober 2026 (Pertemuan 8 - Daring Pemadatan)

## 📁 Struktur Berkas

```text
bab-08-support-vector-machine-svm/
├── main.py                                      # Skrip executable Python mandiri
├── bab_08_bab_08_support_vector_machine_svm.ipynb # Jupyter Notebook terpadu & komprehensif
├── 08_support_vector_machine_svm.ipynb          # Notebook modul utama Unindra
├── 08_alternatif_solar_panel_fault_svm_kernels.ipynb # Notebook studi kasus alternatif
├── LAPORAN_BAB_08_ANGGIT_ARISA_SIWI_PRATAMA_202343570045.pdf # Dokumen Laporan Resmi Cetak
├── laporan_printable.html                       # Template HTML cetak laporan
├── materi/                                      # Arsip materi & lembar modul perkuliahan
│   └── Modul_08_Support_Vector_Machine_SVM.pdf
├── visual_output/                               # Grafik visual hasil komputasi
│   ├── 08_svm_hyperplane_margins.png
│   └── 08_svm_kernel_comparison.png
└── README.md                                    # Dokumentasi modul
```

## 🚀 Panduan Menjalankan

```bash
# Menjalankan skrip simulasi mandiri
python main.py

# Membuka Jupyter Notebook
jupyter notebook bab_08_bab_08_support_vector_machine_svm.ipynb
```

## 📊 Hasil Eksekusi Terminal

```text
[OK] Dataset Solar Panel dimuat: 400 titik pantau modul surya.

==================================================
HASIL EVALUASI SUPPORT VECTOR MACHINE (BAB 08)
==================================================
Koefisien Bobot (w)  : [1.1843, -2.4159]
Bias (b)             : -3.4723
Norma Vektor ||w||   : 2.6906
Lebar Margin (2/||w||): 0.7433
Jumlah Support Vector: 35 titik
Akurasi Kernel LINEAR: 98.00%
Akurasi Kernel RBF   : 99.00%
Akurasi Kernel POLY  : 99.00%
==================================================
[OK] Visual output Bab 08 berhasil disimpan di visual_output/
```

## 💡 Kesimpulan Analitis

1. **Maximal Margin Geometris**: SVM berhasil menemukan bidang optimal dengan lebar margin $0.7433$ unit terstandarisasi yang memaksimalkan separasi antar kelas.
2. **Kekokohan Support Vectors**: Hanya 35 sampel support vector terluar yang menentukan letak hyperplane; titik data lainnya tidak mempengaruhi batas keputusan model.
3. **Keunggulan Kernel Non-Linear**: Fungsi Kernel Gaussian RBF membuktikan akurasi 99.00% pada dinamika non-linier antara suhu dan voltase panel surya.
