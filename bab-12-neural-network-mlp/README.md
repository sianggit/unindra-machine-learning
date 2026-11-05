# Bab 12 - Neural Network dan Multi-Layer Perceptron (MLP)

> **Portofolio Tugas dan Praktikum Machine Learning Unindra**  
> Mahasiswa: **Anggit Arisa Siwi Pratama** | NPM: **`202343570045`** | Kelas: **XC Ekstensi**  
> Dosen: **Nurfidah Dwitiyanti, M.Si.** | Semester Gasal 2026/2027  

---

## Ringkasan Teknis

- **Topik Utama**: Artificial Neural Network (ANN), Multi-Layer Perceptron (MLP), Forward Propagation, Fungsi Aktivasi Non-linear (ReLU, Sigmoid, Softmax), Komputasi Cross-Entropy Loss, Gradient Descent, Backpropagation, dan Kurva Konvergensi Loss.
- **Dataset**: `Scikit-Learn Digits Recognition (N=1.797 sampel, 10 kelas target) dan California Housing Spatial Data`
- **Jadwal Pelaksanaan**: 21 November 2026

## Struktur Berkas

```text
bab-12-neural-network-mlp/
├── main.py                                      # Skrip executable Python mandiri
├── 12_neural_network_mlp.ipynb                  # Notebook modul utama Unindra
├── 12_alternatif_california_housing_mlp_neural_network.ipynb # Notebook studi kasus alternatif
├── LAPORAN_BAB_12_ANGGIT_ARISA_SIWI_PRATAMA_202343570045.pdf # Dokumen Laporan Resmi Cetak A4
├── visual_output/                               # Grafik visual hasil komputasi
│   └── 12_mlp_neural_network_loss_curve.png
└── README.md                                    # Dokumentasi modul
```

## Panduan Menjalankan

```bash
# Menjalankan skrip simulasi mandiri
python main.py

# Membuka Jupyter Notebook
jupyter notebook 12_neural_network_mlp.ipynb
```

## Hasil Eksekusi Terminal

```text
=======================================================
HASIL EVALUASI MULTILAYER PERCEPTRON (BAB 12)
=======================================================
Forward Pass Satu Neuron (Hal 72):
- Input x : [0.5, -0.2, 0.1] | Bobot w: [0.4, 0.3, -0.5]
- Nilai Linier z = (w . x + b) : 0.1900
- Aktivasi Non-Linier ReLU(z)   : 0.1900
Arsitektur Jaringan MLP         : Input (64) -> Dense(64) -> Dense(32) -> Output(10)
Akurasi Uji (Test Accuracy)     : 97.78%
Nilai Loss Akhir (Cross-Entropy): 0.0482
Jumlah Epoch Konvergensi        : 45 iterasi
=======================================================

[OK] Grafik visualisasi disimpan ke visual_output/12_mlp_neural_network_loss_curve.png
```

---
*Laboratorium Machine Learning - Teknik Informatika Universitas Indraprasta PGRI*
