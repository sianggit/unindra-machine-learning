# Bab 14: Mini Project End-to-End: Komparasi Model, Tuning, dan Deployment

> **Portofolio Tugas & Praktikum Machine Learning Unindra**  
> Mahasiswa: **Anggit Arisa Siwi Pratama** | NPM: **`202343570045`** | Kelas: **XC Ekstensi**  
> Dosen Pengampu: **Nurfidah Dwitiyanti, M.Si.** | Semester Gasal 2026/2027  
> Program Studi: **Teknik Informatika (FTIK) - Universitas Indraprasta PGRI**

---

## Ringkasan Teknis Modul

- **Fokus Silabus**: Siklus Penuh CRISP-DM, Komparasi Model Standar Industri, Hyperparameter Tuning Sistematis (`RandomizedSearchCV`), Evaluasi Keputusan Multi-Kriteria, Serialisasi Model (`Joblib`), serta Produksi Web Service (`Streamlit` & `Vercel Next.js Simulator`).
- **Studi Kasus Utama**: *Loan Default Risk Prediction* (Prediksi Risiko Kredit Gagal Bayar Perbankan).
- **Studi Kasus Alternatif**: *Sistem Rekomendasi Kelayakan Beasiswa Mahasiswa Kampus Unindra*.
- **Metode Pembobotan Multi-Kriteria (Sub-bab 14.9)**:
  $$\text{Skor Gabungan} = 0.70 \times F_1\text{-Score (Test)} + 0.30 \times \left(\frac{\text{Interpretability}}{5}\right)$$

---

## Struktur Berkas Terstandarisasi

```text
bab-14-model-deployment-dan-mini-project/
├── main.py                                      # Skrip otomatisasi terminal end-to-end
├── app.py                                       # Antarmuka web produksi Streamlit (Loan Default)
├── app_beasiswa.py                              # Antarmuka web Streamlit kasus alternatif (Beasiswa)
├── requirements.txt                             # Pustaka terisolasi untuk deploy Streamlit Cloud
├── 14_model_deployment_dan_mini_project.ipynb   # Notebook modul utama Unindra (Full CRISP-DM)
├── 14_alternatif_unindra_scholarship_deployment.ipynb # Notebook kasus alternatif mahasiswa
├── LAPORAN_BAB_14_ANGGIT_ARISA_SIWI_PRATAMA_202343570045.pdf # Dokumen Laporan Resmi Cetak A4
├── laporan_printable.html                       # Berkas HTML cetak laporan resmi
├── README.md                                    # Dokumentasi komprehensif modul
├── data/                                        # Dataset mentah & metadata inferensi
│   ├── loan_default_dataset.csv
│   ├── loan_default_weights.json
│   └── unindra_scholarship_applicants.csv
├── models/                                      # Artefak model biner tersimpan
│   ├── model_loan_default_best.joblib
│   └── pipeline_beasiswa_unindra.joblib
├── materi/                                      # Arsip materi resmi modul perkuliahan
│   └── ringkasan_materi_bab_14.md
└── visual_output/                               # Visualisasi analitis 300 DPI
    ├── 14_model_f1_benchmark_comparison.png
    ├── 14_multicriteria_decision_radar.png
    ├── 14_best_model_confusion_matrix.png
    ├── 14_best_model_roc_pr_curves.png
    └── 14_feature_importance_analysis.png
```

---

## Panduan Menjalankan

### 1. Menjalankan Otomasi Pipeline Komputasi & Evaluasi
```bash
python main.py
```

### 2. Menjalankan Aplikasi Web Produksi Secara Lokal

```bash
# Menjalankan antarmuka Streamlit
streamlit run app.py

# Menjalankan REST API & Form Web Flask
python flask_app.py

# Menjalankan pengujian otomatis suite deployment
python test_deployment_apps.py
```

### 3. Mengakses Live Cloud Deployment

- **Streamlit Community Cloud**: [https://ml-lab-anggitp.streamlit.app/](https://ml-lab-anggitp.streamlit.app/)
- **Vercel Interactive Lab**: [https://ml-lab-anggitp.vercel.app/ml-lab/14-model-deployment-dan-mini-project](https://ml-lab-anggitp.vercel.app/ml-lab/14-model-deployment-dan-mini-project)

---

## Hasil Eksekusi Komparasi Model Terminal (`main.py`)

```text
=================================================================
[BAB 14] Mini Project End-to-End: Komparasi Model dan Deployment
   Laboratorium Machine Learning - Universitas Indraprasta PGRI
   Mahasiswa: Anggit Arisa Siwi Pratama (NPM: 202343570045)
=================================================================

[Fase 1: Data Understanding]
- Dataset Kredit tersimpan di: loan_default_dataset.csv
- Dimensi Data: 1200 baris, 7 kolom
- Distribusi Kelas Target (status_default):
  * Lancar (0)      : 1044 (87.0%)
  * Gagal Bayar (1) : 156 (13.0%)

[Fase 2: Data Preparation]
- Data Latih : 960 sampel (Stratified 80%)
- Data Uji   : 240 sampel (Stratified 20%)

[Fase 3: Modeling & Tuning 3 Algoritma]
[Proses] Melatih dan melakukan tuning: Random Forest...
   - CV F1-Score Terbaik : 0.7235
   - Test F1-Score       : 0.6970 | Recall: 0.7419 | PR-AUC: 0.6766
   - Skor Gabungan       : 0.6679 (Interpretability: 3/5)
   - Waktu Latih: 16.17s | Latensi Inferensi: 0.052 ms/sampel

[Proses] Melatih dan melakukan tuning: Gradient Boosting...
   - CV F1-Score Terbaik : 0.7092
   - Test F1-Score       : 0.6667 | Recall: 0.6452 | PR-AUC: 0.7075
   - Skor Gabungan       : 0.5867 (Interpretability: 2/5)
   - Waktu Latih: 2.71s | Latensi Inferensi: 0.005 ms/sampel

[Proses] Melatih dan melakukan tuning: SVM (RBF Kernel)...
   - CV F1-Score Terbaik : 0.6790
   - Test F1-Score       : 0.6575 | Recall: 0.7742 | PR-AUC: 0.5646
   - Skor Gabungan       : 0.5803 (Interpretability: 2/5)
   - Waktu Latih: 1.13s | Latensi Inferensi: 0.037 ms/sampel

[Fase 4: Rekapitulasi Evaluasi & Pemilihan Model]
            Model  CV F1 Test F1 Recall ROC-AUC Latensi (ms)  Interp (1-5) Skor Gabungan
    Random Forest 72.35%  69.70% 74.19%  0.9379        0.052             3        0.6679
Gradient Boosting 70.92%  66.67% 64.52%  0.8924        0.005             2        0.5867
 SVM (RBF Kernel) 67.90%  65.75% 77.42%  0.8835        0.037             2        0.5803

[Hasil] Model terpilih berdasarkan kriteria multi-evaluasi: Random Forest
   Parameter Terbaik: {'n_estimators': 100, 'min_samples_leaf': 1, 'max_depth': None, 'class_weight': 'balanced'}
   Skor Gabungan    : 0.6679

[Fase 5: Serialisasi Artefak Model & Integrasi Vercel]
- Artefak Joblib tersimpan di: models/model_loan_default_best.joblib
- Uji Coba Pemuatan Artefak Sukses: Prediksi Kelas 0 (Probabilitas Default: 3.0%)
- Metadata inferensi Vercel tersimpan di: loan_default_weights.json

[Fase 6: Pembuatan Grafik Analitis Resolusi Tinggi (300 DPI)]
- Grafik 1 tersimpan: 14_model_f1_benchmark_comparison.png
- Grafik 2 tersimpan: 14_multicriteria_decision_radar.png
- Grafik 3 tersimpan: 14_best_model_confusion_matrix.png
- Grafik 4 tersimpan: 14_best_model_roc_pr_curves.png
- Grafik 5 tersimpan: 14_feature_importance_analysis.png

=================================================================
[Selesai] Pipeline Bab 14 berhasil dieksekusi secara lengkap.
=================================================================
```

---

## Interpretasi Bisnis dan Kebijakan Perbankan

1. **Trade-off Sensitivitas vs Akurasi**: Pada portofolio kredit, metrik akurasi 87% dapat dicapai hanya dengan memprediksi seluruh nasabah lancar (*majority class guessing*). Oleh karena itu, pengoptimalan difokuskan pada **Recall (74.19%)** dan **F1-Score (69.70%)** guna menekan kebocoran debitur bermasalah (*False Negative*).
2. **Keunggulan Skor Gabungan Multi-Kriteria**: Random Forest terpilih mengungguli Gradient Boosting karena mampu menghasilkan transparansi fitur (*feature importance*) yang mudah dijelaskan (*interpretable*) kepada komite kredit dan auditor perbankan.
3. **Faktor Risiko Utama**: Dua variabel paling berpengaruh adalah `rasio_cicilan` (*Debt Service Ratio*) dan `skor_kredit` (SLIK OJK), menegaskan bahwa likuiditas bulanan nasabah lebih menentukan kelancaran angsuran dibanding nilai agunan statis semata.
