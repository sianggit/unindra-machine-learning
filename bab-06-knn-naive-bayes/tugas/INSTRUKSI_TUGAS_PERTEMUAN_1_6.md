# 📋 Tugas Machine Learning Pertemuan 1 - 6 (Kisi-Kisi UTS)
**Mata Kuliah**: Machine Learning *) #) (Kelas XC)  
**Dosen Pengampu**: Nurfidah Dwitiyanti  
**Sumber**: LMS UNINDRA  
**File Soal Asli**: [`./Tugas_Machine_Learning_Pertemuan_1-6_Soal.pdf`](./Tugas_Machine_Learning_Pertemuan_1-6_Soal.pdf)

---

## 📌 Ketentuan dan Batas Waktu Pengumpulan
- **Waktu Mulai**: 16 Oktober 2026, 20:39:00 WIB
- **Batas Akhir Submit**: 22 Oktober 2026, 23:59:00 WIB
- **Format File**: 1 Berkas PDF (Hasil scan/foto tulisan tangan)

### ⚠️ Aturan Pengerjaan dari Dosen:
1. **Individu**: Tugas dikerjakan secara individu. Jawaban yang sama persis antarmahasiswa dinilai nol untuk semua pihak.
2. **Keterlambatan**: Pengumpulan setelah batas waktu tidak diterima, kecuali ada alasan yang sah dan disampaikan sebelum batas waktu.
3. **Diskusi**: Pertanyaan tentang soal diajukan di WAG agar jawabannya bisa dibaca semua mahasiswa.
4. **Format Jawaban**: Tulis jawaban tangan di kertas folio atau HVS. Cantumkan rumus dan langkah hitung, bukan hanya hasil akhir. Bulatkan hasil ke 3 desimal.
5. **Format Unggah**: Pindai atau foto jawaban dengan jelas, lalu gabungkan menjadi satu berkas PDF.

---

## 📝 Ringkasan Soal Tugas (Kisi-Kisi UTS)

### Soal 1: Studi Kasus KirimCepat & Workflow ML (CRISP-DM & Evaluasi)
- **Konteks**: 10.000 data pengiriman (8.800 tepat waktu, 1.200 terlambat). Kelas positif = terlambat.
- **1a**: Tentukan kategori ML, jenis tugas, target, metrik utama; hitung akurasi baseline (tebak semua tepat waktu) & jelaskan kelemahannya.
- **1b**: Fase CRISP-DM yang harus dikunjungi kembali untuk: (1) definisi operasional keterlambatan belum fix, (2) data leakage kolom `tanggal_diterima`.
- **1c**: Pembagian data stratified 70 : 10 : 20 (train : val : test). Hitung jumlah sampel tiap kelas & tuliskan fungsi tiap subset.
- **1d**: Analisis gap F1 train vs validation model P, Q, R; diagnosis kondisi model (underfitting/overfitting/good fit), rekomendasi perbaikan, & pemilihan model terbaik.

### Soal 2: Preprocessing, Imputasi & Penanganan Imbalanced Data
- **2a**: Sampel berat paket (kg): `1, 2, 2, 3, 3, 4, 5, 18`. Hitung Q1, Q3, IQR, batas outlier (1.5 IQR), identifikasi kandidat outlier, & pertimbangan pembersihan.
- **2b**: Jarak pengiriman: `5, 8, 10, 12, 65` dan satu nilai kosong. Hitung imputasi mean vs median. Hitung Z-score untuk jarak 55 km ($\mu=15, \sigma=12$) & cek $|z| > 3$.
- **2c**: Perhitungan class weight $w_k = N / (K \times n_k)$ untuk data train Soal 1c dan interpretasi bobotnya.
- **2d**: Interpolasi SMOTE: $x_i = (2, 7)$ dan $x_j = (8, 4)$ dengan $\lambda = 1/3$. Jelaskan mengapa SMOTE hanya diaplikasikan pada data train.

### Soal 3: Feature Engineering & ColumnTransformer
- **3a**: Uji independensi Chi-Square ($\chi^2 = \sum \frac{(O-E)^2}{E}$) fitur "layanan ekonomi" vs target keterlambatan ($\alpha = 0.05, df = 1, \text{kritis} = 3.84$).
- **3b**: Rancang arsitektur `ColumnTransformer` scikit-learn untuk kolom numerik, nominal, dan ordinal; usulan 2 fitur turunan baru; serta penentuan kolom yang harus dibuang untuk mencegah data leakage.

### Soal 4: Decision Tree (CART, ID3 & Pruning)
- **4a**: Node induk (6 terlambat, 14 tepat waktu). Hitung Gini impurity induk dan Gini tertimbang kandidat split X (cuaca buruk) vs Y (layanan ekonomi). Tentukan pilihan CART.
- **4b**: Hitung menggunakan Entropy dan Information Gain (ID3). Bandingkan konsistensi pemilihan split dengan CART.
- **4c**: Cost-Complexity Pruning: $C_\alpha(T) = R(T) + \alpha \times |T|$. Tentukan pohon terpilih (T1, T2, T3) untuk $\alpha = 0.004$ dan $\alpha = 0.025$. Jelaskan pengaruh kenaikan $\alpha$ terhadap bias-variance.
- **4d**: Analisis validation curve `max_depth` (1 s.d. 10). Tentukan kedalaman optimal dan jelaskan fenomena underfitting (depth 1) dan overfitting (depth 10).

### Soal 5: K-Nearest Neighbors (KNN Manual & Optimasi Hyperparameter)
- **5a**: Titik baru $q = (3, 5)$ terhadap 5 titik data latih $D_1 - D_5$. Hitung jarak Euclidean manual, tentukan voting mayoritas $k=3$ vs weighted KNN ($w = 1/d$).
- **5b**: Ulangi kalkulasi menggunakan jarak Manhattan ($L_1$). Analisis apakah tetangga terdekat dan kelas prediksi mengalami perubahan.
- **5c**: Analisis kurva 5-fold cross-validation terhadap nilai $k$ (1 s.d. 21). Tentukan $k$ optimal dan jelaskan mengapa $k$ tidak dipilih berdasarkan akurasi data train.
- **5d**: Urgensi Feature Scaling (StandardScaler/MinMaxScaler) sebelum kalkulasi metrik jarak KNN.
