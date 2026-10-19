# Laboratorium Machine Learning — Universitas Indraprasta PGRI
### Repositori Portofolio Tugas & Praktikum Terkurasi (Semester Gasal 2026/2027)

| Atribut Akademik | Keterangan Resmi |
| :--- | :--- |
| **Nama Mahasiswa** | **Anggit Arisa Siwi Pratama** |
| **NPM** | **`202343570045`** |
| **Program Studi / Kelas** | Teknik Informatika (S1) / **XC Ekstensi** |
| **Mata Kuliah** | **Machine Learning *) #)** (Kode: `KK43K780`) |
| **Dosen Pengampu** | **Nurfidah Dwitiyanti, M.Kom.** |
| **Institusi** | Fakultas Teknik & Ilmu Komputer, **Universitas Indraprasta PGRI (Unindra)** |
| **Target Repository** | [github.com/sianggit/unindra-machine-learning](https://github.com/sianggit/unindra-machine-learning) |

---

## 📌 Ringkasan Portofolio Modul Terlaksana (Pertemuan 1 – 6 & UTS)

Repositori ini berisi kurasi berkas penugasan dan praktikum komputasional yang telah diselesaikan hingga **19 Oktober 2026**. Seluruh modul dilengkapi skrip mandiri Python (`main.py`), Jupyter Notebook interaktif (`.ipynb`), grafik visual resolusi tinggi (`visual_output/`), serta laporan akademik terformat (`.pdf`).

| Direktori Modul | Topik Bahasan & Fokus Algoritma | Dataset Referensi | Status Pengumpulan |
| :--- | :--- | :--- | :---: |
| [`bab-01-pengantar-ml`](./bab-01-pengantar-ml/) | NumPy Vectorization, Z-Score Standardizer & Matriks Kovariansi | Palmer Archipelago Penguins | **Lengkap (.py, .ipynb, .pdf)** |
| [`bab-02-workflow-ml`](./bab-02-workflow-ml/) | CRISP-DM Workflow, Stratified K-Fold vs Random Holdout | IBM HR Employee Attrition | **Lengkap (.py, .ipynb, .pdf)** |
| [`bab-03-preprocessing-missing-imbalanced`](./bab-03-preprocessing-missing-imbalanced/) | Missing Value Imputation (Median) & SMOTE Resampling | Financial Fraud Transactions | **Lengkap (.py, .ipynb, .pdf)** |
| [`bab-04-feature-engineering-pipeline`](./bab-04-feature-engineering-pipeline/) | ColumnTransformer 3-Path & Pipeline Titanic (LR vs RF) | Titanic Manifest (N=891) | **Lengkap (.py, .ipynb, .pdf)** |
| [`bab-05-decision-tree`](./bab-05-decision-tree/) | Pohon Keputusan CART & C4.5, Gini Impurity, Alpha Pruning | UMKM Microloan Applications | **Lengkap (.py, .ipynb, .pdf)** |
| [`bab-06-knn-naive-bayes`](./bab-06-knn-naive-bayes/) | K-NN Distance-Weighted (L1/L2) vs Multinomial Naive Bayes | E-Commerce Sentiment Reviews | **Lengkap (.py, .ipynb, .pdf)** |
| [`tugas-uts-pertemuan-1-6`](./tugas-uts-pertemuan-1-6/) | Solusi Komputasional 5 Paket Soal Studi Kasus KirimCepat | Ekspedisi KirimCepat (10.000 Sampel) | **Siap Nilai (.py, .ipynb, .pdf)** |

---

## 🔒 Kebijakan Isolasi Modul Mendatang (`upcoming/`)

Sesuai dengan etika akademik dan kalender perkuliahan, materi dan praktikum yang **belum dilaksanakan** (Bab 07 s.d. Bab 14) ditempatkan pada direktori `upcoming/` dan **dikecualikan dari pelacakan git** melalui konfigurasi `.gitignore`:

- **Bab 07**: Evaluasi Model Klasifikasi Komprehensif (Jadwal: 30 Oktober 2026)
- **Bab 08**: Support Vector Machine / SVM (Jadwal: 06 November 2026)
- **Bab 09**: Ensemble Learning & Random Forest (Jadwal: 13 November 2026)
- **Bab 10**: Clustering K-Means & DBSCAN (Jadwal: 20 November 2026)
- **Bab 11**: Dimensionality Reduction PCA & t-SNE (Jadwal: 27 November 2026)
- **Bab 12**: Neural Network & Multi-Layer Perceptron (Jadwal: 04 Desember 2026)
- **Bab 13**: Convolutional Neural Network / CNN (Jadwal: 11 Desember 2026)
- **Bab 14**: Model Deployment & Capstone Project (Jadwal: 18 Desember 2026)

Setiap bab di atas akan dibuka dan dikomit secara bertahap tepat setelah perkuliahan tatap muka bab terkait dilaksanakan.

---

## 🛠️ Panduan Replikasi & Lingkungan Eksekusi

### 1. Kebutuhan Lingkungan
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

---
*Dokumentasi disusun mengikuti panduan anti-slop: bebas kalimat klise AI, terverifikasi secara matematis, dan berorientasi pada reproduktibilitas kode murni.*
