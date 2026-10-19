# Tugas Praktikum Pertemuan 1 - 6 (Kisi-Kisi UTS) — Ekspedisi KirimCepat

> **Portofolio Tugas Akademik Machine Learning — Universitas Indraprasta PGRI (Unindra)**  
> **Nama Mahasiswa**: Anggit Arisa Siwi Pratama  
> **NPM**: `202343570045`  
> **Kelas / Prodi**: XC Ekstensi / Teknik Informatika (S1)  
> **Mata Kuliah**: Machine Learning (KK43K780)  
> **Dosen Pengampu**: Nurfidah Dwitiyanti, M.Kom.  
> **Batas Waktu Pengumpulan**: 22 Oktober 2026, 23:59 WIB  

---

## 📌 Deskripsi Tugas & Studi Kasus

Dokumentasi ini memuat penyelesaian komprehensif atas 5 butir soal komputasional dan teoritis berdasarkan studi kasus operasional logistik **Ekspedisi KirimCepat** (dataset 10.000 transaksi pengiriman, rasio ketepatan 88% tepat waktu vs 12% terlambat).

Penyelesaian disajikan dalam tiga format terintegrasi:
1. **Laporan Resmi PDF**: [`LAPORAN_TUGAS_UTS_KIRIMCEPAT_ANGGIT_ARISA_SIWI_PRATAMA_202343570045.pdf`](./LAPORAN_TUGAS_UTS_KIRIMCEPAT_ANGGIT_ARISA_SIWI_PRATAMA_202343570045.pdf)
2. **Jupyter Notebook**: [`tugas_uts_kirimcepat.ipynb`](./tugas_uts_kirimcepat.ipynb)
3. **Skrip Python Mandiri**: [`main.py`](./main.py)
4. **Analisis Tekstual Manual**: [`LEMBAR_JAWABAN_ANALITIS_UTS.md`](./LEMBAR_JAWABAN_ANALITIS_UTS.md)

---

## 📋 Ringkasan Butir Soal & Temuan Algoritmik

| No | Pokok Bahasan & Algoritma | Nilai Parameter / Kasus Uji | Hasil Komputasi | Kesimpulan Akademik |
| :-: | :--- | :--- | :--- | :--- |
| **1** | Evaluasi Metrik & Paradoks Akurasi | $N=10.000$ ($8.800$ tepat, $1.200$ lambat) | Baseline Akurasi: **88,00%**, Recall: **0,00%** | Akurasi menyesatkan pada data timpang; F1-Score & Cost Matrix wajib digunakan. |
| **2** | Deteksi Outlier IQR & Sintesis SMOTE | Paket: $[1, 2, 2, 3, 3, 4, 5, 18]$ kg; $\lambda = 1/3$ | $IQR=2,0$, Batas Atas = **6,5 kg**, Outlier = **18 kg**; Titik Baru = **$(4.0, 6.0)$** | Outlier ekstrem terdeteksi; interpolasi geometris SMOTE berhasil menyeimbangkan kelas. |
| **3** | Seleksi Fitur Uji Asosiasi Chi-Square | Kontingensi $2 \times 2$ (Cuaca Hujan vs Keterlambatan) | $\chi^2_{\text{hitung}} = \mathbf{12,192} > \chi^2_{\text{tabel}} (3,841)$ | Tolak $H_0$ ($p < 0,001$); fitur cuaca signifikan secara statistik dan wajib dipertahankan. |
| **4** | Pemisahan Pohon Keputusan (CART vs ID3) | $D=20$ sampel; Fitur Cuaca (Hujan vs Cerah) | $\text{Gini}_{\text{induk}} = 0,420 \to \text{Gini}_{\text{split}} = \mathbf{0,287}$ | Penurunan Impurity $\Delta \text{Gini} = 0,133$; CART & ID3 konsisten memilih split ini. |
| **5** | Klasifikasi Manual Jarak K-NN | Query $q(3, 5)$ terhadap $D_1 \dots D_5$; $k=3$ | Jarak: $D_2=1,0$, $D_3=1,414$, $D_1=3,606$ | 3 Tetangga Terdekat didominasi kelas terlambat; prediksi akhir: **Terlambat**. |

---

## 🚀 Panduan Eksekusi

```bash
# Menjalankan simulasi numerik mandiri
python main.py

# Membuka Jupyter Notebook interaktif
jupyter notebook tugas_uts_kirimcepat.ipynb
```

Grafik visual komposit otomatis tersimpan di folder [`visual_output/uts_kirimcepat_summary.png`](./visual_output/uts_kirimcepat_summary.png).
