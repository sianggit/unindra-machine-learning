# Laboratorium Machine Learning — Universitas Indraprasta PGRI
### Repositori Portofolio Tugas & Praktikum (Semester Gasal 2026/2027)

| Atribut Akademik | Keterangan Resmi |
| :--- | :--- |
| **Nama Mahasiswa** | **Anggit Arisa Siwi Pratama** |
| **NPM** | **`202343570045`** |
| **Program Studi / Kelas** | Teknik Informatika (S1) / **XC Ekstensi** |
| **Mata Kuliah** | **Machine Learning** (Kode: `KK43K780`) |
| **Dosen Pengampu** | **Nurfidah Dwitiyanti, M.Kom.** |
| **Institusi** | Fakultas Teknik & Ilmu Komputer, **Universitas Indraprasta PGRI (Unindra)** |
| **Portal Interaktif (Live Demo)** | **[ml-lab-anggitp.vercel.app](https://ml-lab-anggitp.vercel.app/)** |

---

> 🌐 **Live Playground & Simulasi Interaktif**:  
> Seluruh implementasi model, notebook interaktif berbasis WebAssembly (Pyodide), visualisasi grafik dinamis, serta lembar jawaban analitis tugas UTS dapat dijalankan secara langsung melalui portal web:  
> 👉 **[https://ml-lab-anggitp.vercel.app/](https://ml-lab-anggitp.vercel.app/)**

---

## 📌 Ringkasan Modul & Hasil Praktikum

Repositori ini memuat berkas penugasan dan praktikum komputasional yang telah dilaksanakan. Seluruh modul dilengkapi skrip mandiri Python (`main.py`), Jupyter Notebook interaktif (`.ipynb`), grafik visualisasi model (`visual_output/`), serta dokumen laporan cetak resmi (`.pdf`).

| Direktori Modul | Topik Bahasan & Fokus Algoritma | Dataset Acuan | Dokumen Laporan Resmi |
| :--- | :--- | :--- | :---: |
| [`bab-01-pengantar-ml`](./bab-01-pengantar-ml/) | NumPy Vectorization, Z-Score Standardizer & Matriks Kovariansi | Palmer Archipelago Penguins | [Laporan Bab 01 (PDF)](./bab-01-pengantar-ml/LAPORAN_BAB_01_ANGGIT_ARISA_SIWI_PRATAMA_202343570045.pdf) |
| [`bab-02-workflow-ml`](./bab-02-workflow-ml/) | CRISP-DM Workflow, Stratified K-Fold vs Random Holdout | IBM HR Employee Attrition | [Laporan Bab 02 (PDF)](./bab-02-workflow-ml/LAPORAN_BAB_02_ANGGIT_ARISA_SIWI_PRATAMA_202343570045.pdf) |
| [`bab-03-preprocessing-missing-imbalanced`](./bab-03-preprocessing-missing-imbalanced/) | Missing Value Imputation (Median) & SMOTE Resampling | Financial Fraud Transactions | [Laporan Bab 03 (PDF)](./bab-03-preprocessing-missing-imbalanced/LAPORAN_BAB_03_ANGGIT_ARISA_SIWI_PRATAMA_202343570045.pdf) |
| [`bab-04-feature-engineering-pipeline`](./bab-04-feature-engineering-pipeline/) | ColumnTransformer 3-Path & Pipeline Titanic (LR vs RF) | Titanic Manifest (N=891) | [Laporan Bab 04 (PDF)](./bab-04-feature-engineering-pipeline/LAPORAN_BAB_04_ANGGIT_ARISA_SIWI_PRATAMA_202343570045.pdf) |
| [`bab-05-decision-tree`](./bab-05-decision-tree/) | Pohon Keputusan CART & C4.5, Gini Impurity, Alpha Pruning | UMKM Microloan Applications | [Laporan Bab 05 (PDF)](./bab-05-decision-tree/LAPORAN_BAB_05_ANGGIT_ARISA_SIWI_PRATAMA_202343570045.pdf) |
| [`bab-06-knn-naive-bayes`](./bab-06-knn-naive-bayes/) | K-NN Distance-Weighted (L1/L2) vs Multinomial Naive Bayes | E-Commerce Sentiment Reviews | [Laporan Bab 06 (PDF)](./bab-06-knn-naive-bayes/LAPORAN_BAB_06_ANGGIT_ARISA_SIWI_PRATAMA_202343570045.pdf) |
| [`tugas-uts-pertemuan-1-6`](./tugas-uts-pertemuan-1-6/) | Solusi Komputasional 5 Paket Soal Studi Kasus KirimCepat | Ekspedisi KirimCepat (10.000 Sampel) | [Laporan Tugas UTS (PDF)](./tugas-uts-pertemuan-1-6/LAPORAN_TUGAS_UTS_KIRIMCEPAT_ANGGIT_ARISA_SIWI_PRATAMA_202343570045.pdf) |

---

## 🛠️ Panduan Replikasi & Lingkungan Eksekusi

### 1. Kebutuhan Pustaka
- Python versi `3.10` atau yang lebih baru.
- Library utama:
  ```bash
  pip install numpy pandas scikit-learn matplotlib
  ```

### 2. Menjalankan Kode Mandiri
Setiap modul dapat langsung dieksekusi melalui terminal:
```bash
# Menjalankan Bab 01
cd bab-01-pengantar-ml
python main.py

# Menjalankan Solusi UTS
cd ../tugas-uts-pertemuan-1-6
python main.py
```

### 3. Membuka Jupyter Notebook
```bash
jupyter notebook
```
