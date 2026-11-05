# Bab 13 - Convolutional Neural Network (CNN) dan Transfer Learning

> **Portofolio Tugas dan Praktikum Machine Learning Unindra**  
> Mahasiswa: **Anggit Arisa Siwi Pratama** | NPM: **`202343570045`** | Kelas: **XC Ekstensi**  
> Dosen: **Nurfidah Dwitiyanti, M.Si.** | Semester Gasal 2026/2027  

---

## Ringkasan Teknis

- **Topik Utama**: Pemrosesan Citra Digital, Ekstraksi Fitur Hierarkis, Operasi Konvolusi 2D dengan Kernel Filter (Deteksi Tepi Vertikal/Horizontal Sobel), Feature Maps, Non-linear Rectified Linear Unit (ReLU), Downsampling Spasial dengan Max Pooling, dan Konsep Transfer Learning.
- **Dataset**: `Pola Poligon Citra Grayscale Sintetis dan Klasifikasi Objek Visual CIFAR-10`
- **Jadwal Pelaksanaan**: 28 November 2026

## Struktur Berkas

```text
bab-13-cnn-dan-transfer-learning/
├── main.py                                      # Skrip executable Python mandiri
├── 13_cnn_dan_transfer_learning.ipynb           # Notebook modul utama Unindra
├── 13_alternatif_cifar10_urban_vision_cnn.ipynb # Notebook studi kasus alternatif
├── LAPORAN_BAB_13_ANGGIT_ARISA_SIWI_PRATAMA_202343570045.pdf # Dokumen Laporan Resmi Cetak A4
├── visual_output/                               # Grafik visual hasil komputasi
│   └── 13_cnn_convolution_filter_feature_map.png
└── README.md                                    # Dokumentasi modul
```

## Panduan Menjalankan

```bash
# Menjalankan skrip simulasi mandiri
python main.py

# Membuka Jupyter Notebook
jupyter notebook 13_cnn_dan_transfer_learning.ipynb
```

## Hasil Eksekusi Terminal

```text
=======================================================
HASIL EVALUASI OPERASI KONVOLUSI 2D & CNN (BAB 13)
=======================================================
Dimensi Citra Input Asli     : 8 x 8 piksel
Dimensi Kernel Filter Sobel  : 3 x 3
Dimensi Feature Map Konvolusi: 6 x 6
Dimensi Setelah Max Pooling  : 3 x 3 (Subsampling 4x)
Respon Tepi Vertikal Maksimum: 4.0 unit aktivasi
=======================================================

[OK] Grafik visualisasi disimpan ke visual_output/13_cnn_convolution_filter_feature_map.png
```

---
*Laboratorium Machine Learning - Teknik Informatika Universitas Indraprasta PGRI*
