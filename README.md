# Laboratorium Machine Learning - Universitas Indraprasta PGRI
### Repositori Portofolio Tugas dan Praktikum (Semester Gasal 2026/2027)

| Atribut Akademik | Keterangan Resmi |
| :--- | :--- |
| **Nama Mahasiswa** | **Anggit Arisa Siwi Pratama** |
| **NPM** | **`202343570045`** |
| **Program Studi / Kelas** | Teknik Informatika (S1) / **XC Ekstensi** |
| **Mata Kuliah** | **Machine Learning** (Kode: `KK43K780`, 3 SKS) |
| **Dosen Pengampu** | **Nurfidah Dwitiyanti, M.Si.** |
| **Institusi** | Fakultas Teknik dan Ilmu Komputer, **Universitas Indraprasta PGRI (Unindra)** |
| **Portal Edukasi Web (Vercel)** | **[ml-lab-anggitp.vercel.app](https://ml-lab-anggitp.vercel.app/)** |
| **Layanan Produksi Streamlit Cloud** | **[unindra-machine-learning-anggitp.streamlit.app](https://unindra-machine-learning-anggitp.streamlit.app/)** |

---

## Ekosistem Digital Laboratorium

Portofolio ini terhubung dengan dua platform produksi aktif:

1. **Portal Edukasi WebAssembly (Vercel Next.js)**: [https://ml-lab-anggitp.vercel.app/](https://ml-lab-anggitp.vercel.app/)  
   Menyediakan kurikulum interaktif 14 bab perkuliahan, simulasi eksekusi kode Python langsung di peramban tanpa backend server via Pyodide WebAssembly, showcase penugasan lengkap, dan generator ekspor laporan PDF.
2. **Aplikasi Produksi Analisis Risiko Kredit (Streamlit Cloud)**: [https://unindra-machine-learning-anggitp.streamlit.app/](https://unindra-machine-learning-anggitp.streamlit.app/)  
   Antarmuka analitis keputusan komite kredit perbankan interaktif (Bab 14) berbasis model ensemble Random Forest terlatih dengan evaluasi keputusan multi-kriteria.

---

## Ringkasan Modul dan Hasil Praktikum

Repositori ini memuat seluruh berkas penugasan dan praktikum komputasional yang telah diselesaikan. Setiap modul dilengkapi skrip mandiri Python (`main.py`), Jupyter Notebook interaktif (`.ipynb`), visualisasi resolusi tinggi (`visual_output/`), serta dokumen laporan cetak resmi format A4 (`.pdf`).

| Direktori Modul | Topik Bahasan dan Fokus Algoritma | Dataset Acuan | Dokumen Laporan Resmi |
| :--- | :--- | :--- | :---: |
| [`bab-01-pengantar-ml`](./bab-01-pengantar-ml/) | NumPy Vectorization, Z-Score Standardizer, dan Matriks Kovariansi | Palmer Archipelago Penguins | [Laporan Bab 01 (PDF)](./bab-01-pengantar-ml/LAPORAN_BAB_01_ANGGIT_ARISA_SIWI_PRATAMA_202343570045.pdf) |
| [`bab-02-workflow-ml`](./bab-02-workflow-ml/) | CRISP-DM Workflow, Stratified K-Fold vs Random Holdout | IBM HR Employee Attrition | [Laporan Bab 02 (PDF)](./bab-02-workflow-ml/LAPORAN_BAB_02_ANGGIT_ARISA_SIWI_PRATAMA_202343570045.pdf) |
| [`bab-03-preprocessing-missing-imbalanced`](./bab-03-preprocessing-missing-imbalanced/) | Missing Value Imputation (Median) dan SMOTE Resampling | Financial Fraud Transactions | [Laporan Bab 03 (PDF)](./bab-03-preprocessing-missing-imbalanced/LAPORAN_BAB_03_ANGGIT_ARISA_SIWI_PRATAMA_202343570045.pdf) |
| [`bab-04-feature-engineering-pipeline`](./bab-04-feature-engineering-pipeline/) | ColumnTransformer 3-Path dan Pipeline Titanic (LR vs RF) | Titanic Manifest (N=891) | [Laporan Bab 04 (PDF)](./bab-04-feature-engineering-pipeline/LAPORAN_BAB_04_ANGGIT_ARISA_SIWI_PRATAMA_202343570045.pdf) |
| [`bab-05-decision-tree`](./bab-05-decision-tree/) | Pohon Keputusan CART dan C4.5, Gini Impurity, Alpha Pruning | UMKM Microloan Applications | [Laporan Bab 05 (PDF)](./bab-05-decision-tree/LAPORAN_BAB_05_ANGGIT_ARISA_SIWI_PRATAMA_202343570045.pdf) |
| [`bab-06-knn-naive-bayes`](./bab-06-knn-naive-bayes/) | K-NN Distance-Weighted (L1/L2) vs Multinomial Naive Bayes | E-Commerce Sentiment Reviews | [Laporan Bab 06 (PDF)](./bab-06-knn-naive-bayes/LAPORAN_BAB_06_ANGGIT_ARISA_SIWI_PRATAMA_202343570045.pdf) |
| [`tugas-uts-pertemuan-1-6`](./tugas-uts-pertemuan-1-6/) | Solusi Komputasional 5 Paket Soal Kasus Ekspedisi KirimCepat | Ekspedisi KirimCepat (10.000 Sampel) | [Laporan Tugas UTS (PDF)](./tugas-uts-pertemuan-1-6/LAPORAN_TUGAS_UTS_KIRIMCEPAT_ANGGIT_ARISA_SIWI_PRATAMA_202343570045.pdf) |
| [`bab-07-evaluasi-model-klasifikasi`](./bab-07-evaluasi-model-klasifikasi/) | Matriks Konfusi, Presisi, Recall, F1-Score, ROC-AUC, Stratified K-Fold | Cyber Intrusion Logs (N=1.000) | [Laporan Bab 07 (PDF)](./bab-07-evaluasi-model-klasifikasi/LAPORAN_BAB_07_ANGGIT_ARISA_SIWI_PRATAMA_202343570045.pdf) |
| [`bab-08-support-vector-machine-svm`](./bab-08-support-vector-machine-svm/) | Maximal Margin Hyperplane, Support Vectors, Komparasi Kernel (RBF/Linear) | Solar Panel Faults (N=400) | [Laporan Bab 08 (PDF)](./bab-08-support-vector-machine-svm/LAPORAN_BAB_08_ANGGIT_ARISA_SIWI_PRATAMA_202343570045.pdf) |
| [`bab-09-ensemble-learning-dan-rf`](./bab-09-ensemble-learning-dan-rf/) | Ensemble Bagging (Random Forest) vs Boosting, Out-of-Bag (OOB) Score | Telco Churn Customers (N=600) | [Laporan Bab 09 (PDF)](./bab-09-ensemble-learning-dan-rf/LAPORAN_BAB_09_ANGGIT_ARISA_SIWI_PRATAMA_202343570045.pdf) |
| [`bab-10-clustering-kmeans-dan-dbscan`](./bab-10-clustering-kmeans-dan-dbscan/) | K-Means Clustering, Evaluasi Elbow dan Silhouette, DBSCAN Density | Spotify Audio Track Features (N=800) | [Laporan Bab 10 (PDF)](./bab-10-clustering-kmeans-dan-dbscan/LAPORAN_BAB_10_ANGGIT_ARISA_SIWI_PRATAMA_202343570045.pdf) |
| [`bab-11-dimensionality-reduction-pca-tsne`](./bab-11-dimensionality-reduction-pca-tsne/) | Reduksi Dimensi Linear (PCA) vs Manifold Non-linear (t-SNE) | Olivetti High-Dimensional Faces (4.096 Dim) | [Laporan Bab 11 (PDF)](./bab-11-dimensionality-reduction-pca-tsne/LAPORAN_BAB_11_ANGGIT_ARISA_SIWI_PRATAMA_202343570045.pdf) |
| [`bab-12-neural-network-mlp`](./bab-12-neural-network-mlp/) | Multi-Layer Perceptron (MLP), Backpropagation, Kurva Konvergensi Loss | California Housing Spatial Regression (N=1.000) | [Laporan Bab 12 (PDF)](./bab-12-neural-network-mlp/LAPORAN_BAB_12_ANGGIT_ARISA_SIWI_PRATAMA_202343570045.pdf) |
| [`bab-13-cnn-dan-transfer-learning`](./bab-13-cnn-dan-transfer-learning/) | Konvolusi 2D, Feature Maps, Pola Kernel Spasial, Visualisasi Filter | Urban Vision Object Classification (CIFAR-10) | [Laporan Bab 13 (PDF)](./bab-13-cnn-dan-transfer-learning/LAPORAN_BAB_13_ANGGIT_ARISA_SIWI_PRATAMA_202343570045.pdf) |
| [`bab-14-model-deployment-dan-mini-project`](./bab-14-model-deployment-dan-mini-project/) | Mini Project CRISP-DM, Keputusan Multi-Kriteria, Dual Deployment (Streamlit & Flask) | Loan Default Risk (N=1.200) | [Laporan Bab 14 (PDF)](./bab-14-model-deployment-dan-mini-project/LAPORAN_BAB_14_ANGGIT_ARISA_SIWI_PRATAMA_202343570045.pdf) |

---

## Panduan Replikasi dan Lingkungan Eksekusi

### 1. Kebutuhan Pustaka
- Python versi `3.10` atau yang lebih baru.
- Instalasi dependensi:
  ```bash
  pip install -r bab-14-model-deployment-dan-mini-project/requirements.txt
  ```

### 2. Menjalankan Skrip Mandiri Modul
Setiap modul dapat langsung dieksekusi melalui terminal untuk mereproduksi komputasi dan visualisasi 300 DPI:
```bash
# Menjalankan Bab 01
cd bab-01-pengantar-ml
python main.py

# Menjalankan Solusi UTS
cd ../tugas-uts-pertemuan-1-6
python main.py

# Menjalankan Pipeline Bab 14
cd ../bab-14-model-deployment-dan-mini-project
python main.py
```

### 3. Menjalankan Aplikasi Web Produksi (Bab 14)
```bash
cd bab-14-model-deployment-dan-mini-project

# Menjalankan antarmuka Streamlit lokal
streamlit run app.py

# Menjalankan REST API & Form Web Flask
python flask_app.py

# Menjalankan pengujian otomatis unit test deployment
python test_deployment_apps.py
```

### 4. Membuka Jupyter Notebook
```bash
jupyter notebook
```
