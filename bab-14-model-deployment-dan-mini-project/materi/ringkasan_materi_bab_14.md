# Ringkasan Materi Resmi: Bab 14: Mini Project End-to-End: Komparasi Model, Tuning, dan Deployment
**Bahan Ajar Mata Kuliah Machine Learning (3 SKS)**  
**Program Studi Teknik Informatika, Fakultas Teknik dan Ilmu Komputer, Universitas Indraprasta PGRI (Unindra)**  
*Penyusun: Nurfidah Dwitiyanti, M.Si. | Cetakan I: Agustus 2026 | UNINDRA PRESS*

---

## 14.1. Struktur Mini Project (Kerangka CRISP-DM)
Proyek akhir praktikum Machine Learning mengimplementasikan siklus penuh **CRISP-DM** (*Cross-Industry Standard Process for Data Mining*):
1. **Business Understanding**: Menentukan masalah domain bisnis secara terukur (contoh: prediksi gagal bayar kredit / loan default) dan menetapkan metrik keberhasilan bisnis.
2. **Data Understanding & Preparation**: Mengumpulkan dataset, melakukan Analisis Data Eksploratif (EDA), menangani nilai hilang (*missing values* via KNN Imputer), mendeteksi outlier, menangani data tidak seimbang (*imbalanced data* via SMOTE / class weight), dan merekayasa fitur (*feature engineering*).
3. **Modeling**: Melatih minimal 3 algoritma berbeda yang telah dipelajari dari Bab 5–13 (contoh: Decision Tree, Random Forest, Gradient Boosting, SVM via Pipeline).
4. **Evaluation**: Membandingkan performa model dengan protokol adil (split konsisten & Stratified 5-Fold Cross-Validation), serta menimbang trade-off antara akurasi, interpretability, dan latensi komputasi.
5. **Deployment**: Membungkus model terbaik ke dalam artefak biner (`.joblib`/`.pkl`) dan merancang antarmuka aplikasi web interaktif (*Streamlit* / *Flask*).

---

## 14.2. Perbandingan Model: Praktik Terbaik
- Menggunakan protokol pengujian yang adil (*fair evaluation protocol*): partisi data latih dan uji yang sama, skema validasi silang berstrata yang sama, dan metrik yang selaras dengan tujuan bisnis (F1-score, Recall, dan PR-AUC).
- Model terbaik tidak selalu model dengan akurasi nominal tertinggi; pertimbangan kepraktisan seperti kemudahan interpretasi (*interpretability*) bagi pemangku kepentingan dan kecepatan waktu respons inferensi sangat menentukan pemilihan akhir.

---

## 14.3. Hyperparameter Tuning Sistematis
- **GridSearchCV**: Mencoba seluruh kombinasi parameter dalam grid secara mendalam. Cocok untuk ruang pencarian kecil-menengah.
- **RandomizedSearchCV**: Mengambil sampel acak dari ruang hyperparameter. Sangat efisien ketika ruang pencarian berdimensi tinggi tanpa membebani komputasi secara berlebih.

---

## 14.4 & 14.5. Deployment: Dari Notebook ke Aplikasi
1. **Serialisasi Model**: Menggunakan `joblib.dump(model, "model_terbaik.pkl")` untuk menyimpan model dan seluruh langkah transformasi data (*pipeline*) agar terhindar dari *data leakage*.
2. **Pemuatan Artefak**: Menggunakan `joblib.load("model_terbaik.pkl")` pada berkas aplikasi web (`app.py`).
3. **Antarmuka Interaktif**: Menggunakan framework modern **Streamlit** untuk menerima input pengguna melalui komponen widget (`st.number_input`, `st.slider`) dan menampilkan prediksi serta probabilitas risiko secara *real-time*.

---

## 14.6. Struktur Laporan Ilmiah
Laporan akhir wajib mengikuti format 6 bagian terstandarisasi:
1. **Pendahuluan**: Latar belakang, rumusan masalah bisnis, dan tujuan proyek.
2. **Tinjauan Data**: Sumber data, deskripsi fitur, dan temuan analisis data awal (EDA).
3. **Metodologi**: Pra-pemrosesan data, konfigurasi algoritma, strategi tuning, dan skema validasi silang.
4. **Hasil dan Pembahasan**: Tabel komparasi performa model, visualisasi metrik (Confusion Matrix, ROC/PR Curves), dan analisis *feature importance*.
5. **Deployment**: Dokumentasi arsitektur sistem dan tangkapan layar aplikasi web yang dibangun.
6. **Kesimpulan**: Ringkasan capaian, batasan model saat ini, dan rekomendasi pengembangan lanjutan.

---

## 14.8 & 14.9. Studi Kasus & Formula Skor Terboboti Multi-Kriteria
- **Studi Kasus**: Prediksi Risiko Gagal Bayar Pinjaman (*Loan Default Prediction*).
- **Rumus Pembobotan Multi-Kriteria**:
  $$\text{Skor Gabungan} = 0.7 \times F_1 + 0.3 \times \left(\frac{\text{Interpretability}}{5}\right)$$
- Skema ini memberikan kompromi optimal antara ketajaman deteksi risiko kelas minoritas (Recall & F1) dengan transparansi regulasi perbankan.

