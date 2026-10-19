# LEMBAR JAWABAN TUGAS PERTEMUAN 1 - 6 (KISI-KISI UTS)
**Mata Kuliah**: Machine Learning *) #) (Kelas XC Ekstensi)  
**Dosen Pengampu**: Nurfidah Dwitiyanti  
**Nama Mahasiswa**: Anggit Arisa Siwi Pratama  
**NPM**: 202343570045  
**Ketentuan**: Ditulis tangan pada kertas Folio bergaris / HVS, seluruh rumus & langkah hitung dicantumkan lengkap, pembulatan hasil akhir ke 3 desimal.

---

## SOAL 1: STUDI KASUS KIRIMCEPAT & WORKFLOW ML

### 1a. Kategori ML, Jenis Tugas, Target, Metrik Utama, dan Evaluasi Baseline
1. **Kategori Machine Learning**: **Supervised Learning** (karena data pelatihan memiliki label historis).
2. **Jenis Tugas (Task)**: **Binary Classification** (Klasifikasi Biner: memprediksi salah satu dari 2 kelas target).
3. **Target Prediksi**: Status keterlambatan pengiriman paket (`terlambat` = 1 sebagai kelas positif, dan `tepat waktu` = 0 sebagai kelas negatif).
4. **Metrik Evaluasi Utama**:
   - **Recall (Sensitivitas)**: Mengukur persentase paket yang benar-benar terlambat yang berhasil dideteksi oleh model.
   - **F1-Score**: Rata-rata harmonik antara Precision dan Recall, sangat tepat untuk data dengan distribusi kelas tidak seimbang (*imbalanced data*).

5. **Perhitungan Akurasi Model Baseline (Menebak Semua Paket "Tepat Waktu")**:
   - Total data ($N$) = $10.000$ paket
   - Paket tepat waktu = $8.800$ paket
   - Paket terlambat = $1.200$ paket
   
   $$\text{Akurasi Baseline} = \frac{\text{Jumlah Tebakan Benar}}{\text{Total Data}} = \frac{8.800}{10.000} = 0{,}880 \text{ (atau } 88{,}000\%)$$

6. **Mengapa Akurasi Tidak Layak Menjadi Metrik Utama?**
   - Terjadi fenomena **Accuracy Paradox**. Meskipun nilai akurasi tampak tinggi ($88{,}000\%$), model tersebut sama sekali tidak memiliki kemampuan prediksi karena memprediksi seluruh paket sebagai "tepat waktu" ($0$ paket terlambat terdeteksi).
   - Akibatnya:
     $$\text{Recall} = \frac{TP}{TP + FN} = \frac{0}{0 + 1.200} = 0{,}000$$
     $$\text{F1-Score} = 0{,}000$$
   - Model gagal mencapai tujuan bisnis KirimCepat, yaitu mendeteksi dini paket yang terlambat guna memberi tahu pelanggan lebih awal.

---

### 1b. Fase CRISP-DM dan Tindakan Perbaikan
1. **Temuan (1): Tim operasional belum menentukan definisi "terlambat" (apakah melewati estimasi 1 hari atau 3 hari)**
   - **Fase CRISP-DM**: **Business Understanding** (Pemahaman Bisnis).
   - **Tindakan Perbaikan**: Mengadakan rapat penyelarasan (*alignment meeting*) dengan tim operasional dan manajemen untuk menetapkan definisi operasional yang baku (Service Level Agreement / SLA), misalnya paket dinyatakan terlambat jika waktu kedatangan $> \text{estimasi} + 24 \text{ jam}$. Definisi ini harus disepakati sebelum proses pelabelan data dilakukan.

2. **Temuan (2): Kolom `tanggal_diterima` ikut dipakai sebagai fitur input**
   - **Fase CRISP-DM**: **Data Preparation** (Persiapan Data).
   - **Tindakan Perbaikan**: Menghapus (*drop*) kolom `tanggal_diterima` dari himpunan fitur input ($X$). Kolom ini merupakan informasi pasca-kejadian (*post-event feature*) yang belum ada saat paket baru dikirim. Menggunakannya menyebabkan kebocoran data (**Data Leakage**), sehingga performa model saat pengujian tampak sempurna namun gagal total saat diterapkan di produksi (*real-time inference*).

---

### 1c. Pembagian Data Stratified (70 : 10 : 20) dan Fungsi Subset
Pembagian bertingkat (*stratified split*) menjaga proporsi kelas tepat waktu ($88{,}000\%$) dan terlambat ($12{,}000\%$) pada setiap himpunan bagian (*subset*):

1. **Training Set (70%)**:
   - Tepat Waktu: $8.800 \times 0{,}70 = 6.160$ paket
   - Terlambat: $1.200 \times 0{,}70 = 840$ paket
   - **Total Train**: $6.160 + 840 = 7.000$ paket
   - **Fungsi**: Digunakan oleh algoritma untuk mempelajari pola data dan mengestimasi bobot/parameter internal model.

2. **Validation Set (10%)**:
   - Tepat Waktu: $8.800 \times 0{,}10 = 880$ paket
   - Terlambat: $1.200 \times 0{,}10 = 120$ paket
   - **Total Validation**: $880 + 120 = 1.000$ paket
   - **Fungsi**: Digunakan untuk penalaan *hyperparameter* (misal: kedalaman pohon, nilai $k$), seleksi arsitektur model, dan mendeteksi gejala overfitting sebelum pengujian akhir.

3. **Test Set (20%)**:
   - Tepat Waktu: $8.800 \times 0{,}20 = 1.760$ paket
   - Terlambat: $1.200 \times 0{,}20 = 240$ paket
   - **Total Test**: $1.760 + 240 = 2.000$ paket
   - **Fungsi**: Digunakan untuk mengukur performa akhir model secara objektif pada data independen yang belum pernah dilihat (*unseen data*).

---

### 1d. Analisis Kinerja Tiga Model (P, Q, R)
Rumus perhitungan selisih (*gap*):
$$\text{Gap} = \text{F1 Train} - \text{F1 Validation}$$

1. **Model P**:
   - $\text{Gap} = 0{,}520 - 0{,}510 = 0{,}010$
   - **Diagnosis**: **Underfitting** (Bias Tinggi). F1 Train dan F1 Validation keduanya sangat rendah ($0{,}520$ dan $0{,}510$), menandakan model terlalu sederhana untuk menangkap kompleksitas data.
   - **Tindakan Perbaikan**: Menambah kompleksitas model (misal: memperdalam pohon keputusan, menggunakan algoritma ensemble seperti Random Forest/XGBoost) atau menambah fitur-fitur baru (*feature engineering*).

2. **Model Q**:
   - $\text{Gap} = 0{,}800 - 0{,}770 = 0{,}030$
   - **Diagnosis**: **Good Fit** (Trade-off Bias-Variance Optimal). F1 Train tinggi ($0{,}800$), F1 Validation tinggi ($0{,}770$), dan selisih kecil ($0{,}030$), menunjukkan generalisasi yang stabil.

3. **Model R**:
   - $\text{Gap} = 0{,}970 - 0{,}650 = 0{,}320$
   - **Diagnosis**: **Overfitting** (Variance Tinggi). F1 Train mendekati sempurna ($0{,}970$) namun F1 Validation anjlok drastis ke $0{,}650$ dengan gap sangat besar ($0{,}320$). Model menghafal derau (*noise*) pada data latih.
   - **Tindakan Perbaikan**: Menerapkan regularisasi (misal: *cost-complexity pruning*, pembatasan `max_depth`, menaikkan `min_samples_split`), atau memperbanyak data latih.

4. **Keputusan Pemilihan Model**:
   - **Pilihan**: **Model Q**.
   - **Alasan**: Model Q memiliki nilai F1 Validation tertinggi ($0{,}770$) dengan tingkat generalisasi yang konsisten dan stabil (gap hanya $0{,}030$).

---

## SOAL 2: PREPROCESSING, IMPUTASI & PENANGANAN IMBALANCED DATA

### 2a. Sampel Berat Paket: Deteksi Outlier dengan IQR
Data berat paket (kg) terurut ($n = 8$):
$$1, \quad 2, \quad 2, \quad 3, \quad 3, \quad 4, \quad 5, \quad 18$$

1. **Langkah Perhitungan Kuartil (Metode Belah Dua / Tukey)**:
   - Median membagi data menjadi dua kelompok:
     - Paruh bawah ($n = 4$): $1, \quad 2, \quad 2, \quad 3$
     - Paruh atas ($n = 4$): $3, \quad 4, \quad 5, \quad 18$
   
   - **Kuartil 1 ($Q_1$)**:
     $$Q_1 = \frac{2 + 2}{2} = 2{,}000 \text{ kg}$$
   
   - **Kuartil 2 / Median ($Q_2$)**:
     $$Q_2 = \frac{3 + 3}{2} = 3{,}000 \text{ kg}$$
   
   - **Kuartil 3 ($Q_3$)**:
     $$Q_3 = \frac{4 + 5}{2} = 4{,}500 \text{ kg}$$

2. **Interquartile Range ($IQR$)**:
   $$IQR = Q_3 - Q_1 = 4{,}500 - 2{,}000 = 2{,}500 \text{ kg}$$

3. **Batas Outlier (Metode Tukey $1{,}5 \times IQR$)**:
   - **Batas Bawah**:
     $$\text{Batas Bawah} = Q_1 - 1{,}5 \times IQR = 2{,}000 - (1{,}5 \times 2{,}500) = 2{,}000 - 3{,}750 = -1{,}750 \text{ kg}$$
   - **Batas Atas**:
     $$\text{Batas Atas} = Q_3 + 1{,}5 \times IQR = 4{,}500 + (1{,}5 \times 2{,}500) = 4{,}500 + 3{,}750 = 8{,}250 \text{ kg}$$

4. **Identifikasi Kandidat Outlier**:
   - Nilai $18 \text{ kg} > \text{Batas Atas } (8{,}250 \text{ kg})$, sehingga **$18 \text{ kg}$ adalah kandidat outlier**.

5. **Pertimbangan Apakah Nilai Tersebut Sebaiknya Dihapus**:
   - **Tidak disarankan langsung dihapus**. Verifikasi terlebih dahulu dengan pihak operasional apakah nilai $18 \text{ kg}$ merupakan kesalahan input (*typo*, misal seharusnya $1{,}8 \text{ kg}$) atau data riil (paket kargo berat). Jika valid secara bisnis, lakukan teknik transformasi logaritma atau *Winsorizing* (capping pada batas atas $8{,}250 \text{ kg}$) agar informasi paket kargo tidak hilang dan model tetap robust.

---

### 2b. Jarak Pengiriman: Imputasi Mean vs Median dan Z-Score
Data jarak yang tersedia ($n = 5$):
$$5, \quad 8, \quad 10, \quad 12, \quad 65$$

1. **Perhitungan Nilai Imputasi**:
   - **Mean (Rata-rata)**:
     $$\text{Mean} = \frac{\sum x}{n} = \frac{5 + 8 + 10 + 12 + 65}{5} = \frac{100}{5} = 20{,}000 \text{ km}$$
   
   - **Median (Nilai Tengah)**:
     - Data terurut: $5, \quad 8, \quad \mathbf{10}, \quad 12, \quad 65$
     $$\text{Median} = 10{,}000 \text{ km}$$

2. **Pemilihan Metode yang Lebih Tepat**:
   - **Pilihan**: **Median ($10{,}000 \text{ km}$)**.
   - **Alasan**: Data memiliki distribusi menceng ke kanan (*skewed*) dengan keberadaan pencilan ekstrem ($65 \text{ km}$). Nilai mean ($20{,}000 \text{ km}$) terdistorsi jauh di atas mayoritas data ($5, 8, 10, 12$), sedangkan median bersifat *robust* (kebal) terhadap pencilan.

3. **Perhitungan Z-Score Paket Berjarak $55 \text{ km}$ ($\mu = 15, \sigma = 12$)**:
   $$z = \frac{x - \mu}{\sigma} = \frac{55 - 15}{12} = \frac{40}{12} \approx 3{,}333$$

4. **Pemeriksaan Kandidat Outlier**:
   $$|z| = |3{,}333| = 3{,}333 > 3{,}000$$
   - **Kesimpulan**: Karena $|z| > 3{,}000$, jarak $55 \text{ km}$ **valid dikategorikan sebagai kandidat outlier**.

---

### 2c. Perhitungan Class Weight ($w_k = \frac{N}{K \times n_k}$)
Diketahui dari subset data train (Soal 1c):
- Total data train ($N$) = $7.000$
- Jumlah kelas ($K$) = $2$
- Jumlah kelas tepat waktu ($n_0$) = $6.160$
- Jumlah kelas terlambat ($n_1$) = $840$

1. **Bobot Kelas Tepat Waktu ($w_0$)**:
   $$w_0 = \frac{N}{K \times n_0} = \frac{7.000}{2 \times 6.160} = \frac{7.000}{12.320} \approx 0{,}568$$

2. **Bobot Kelas Terlambat ($w_1$)**:
   $$w_1 = \frac{N}{K \times n_1} = \frac{7.000}{2 \times 840} = \frac{7.000}{1.680} \approx 4{,}167$$

3. **Arti Perbandingan Bobot**:
   $$\text{Rasio} = \frac{w_1}{w_0} = \frac{4{,}167}{0{,}568} \approx 7{,}336 \quad \left(\text{atau eksak } \frac{6.160}{840} = 7{,}333\right)$$
   - **Interpretasi**: Kelas minoritas (terlambat) diberi bobot penalti sekitar **$7{,}333$ kali lebih besar** dibandingkan kelas mayoritas (tepat waktu) dalam fungsi kerugian (*loss function*). Hal ini memaksa algoritma pembelajaran untuk lebih memprioritaskan pengurangan kesalahan prediksi (*false negative*) pada paket terlambat.

---

### 2d. Sintesis Data SMOTE dan Urgensi Penerapan pada Data Train
Diketahui:
- $x_i = (2, 7)$
- $x_j = (8, 4)$
- $\lambda = \frac{1}{3} \approx 0{,}333$

1. **Perhitungan $x_{\text{baru}}$**:
   $$x_{\text{baru}} = x_i + \lambda (x_j - x_i)$$
   
   - **Komponen Pertama ($x_1$)**:
     $$x_{\text{baru}, 1} = 2 + \frac{1}{3} (8 - 2) = 2 + \frac{1}{3} (6) = 2 + 2{,}000 = 4{,}000$$
   
   - **Komponen Kedua ($x_2$)**:
     $$x_{\text{baru}, 2} = 7 + \frac{1}{3} (4 - 7) = 7 + \frac{1}{3} (-3) = 7 - 1{,}000 = 6{,}000$$
   
   - **Hasil Akhir**:
     $$x_{\text{baru}} = (4{,}000; \quad 6{,}000)$$

2. **Mengapa SMOTE Hanya Diterapkan pada Data Train?**
   - SMOTE bertujuan menyeimbangkan ruang fitur saat proses induksi model pembelajaran. Jika diterapkan pada set validasi atau uji, sampel sintetis akan mencemari data evaluasi (**Data Leakage**). Evaluasi model wajib dilakukan pada data asli yang mencerminkan distribusi populasi riil di lapangan agar estimasi performa tidak mengalami bias optimis palsu (*overly optimistic evaluation*).

---

## SOAL 3: FEATURE ENGINEERING & COLUMNTRANSFORMER

### 3a. Uji Independensi Chi-Square ($\chi^2$) Fitur Layanan Ekonomi
Tabel Kontingensi Observasi ($O$):
| Layanan Ekonomi | Terlambat | Tepat Waktu | Total Baris ($R_i$) |
| :--- | :---: | :---: | :---: |
| **Ya** | $36$ | $64$ | $100$ |
| **Tidak** | $14$ | $86$ | $100$ |
| **Total Kolom ($C_j$)** | $50$ | $150$ | **$N = 200$** |

1. **Perhitungan Frekuensi Harapan ($E_{ij} = \frac{R_i \times C_j}{N}$)**:
   - $E_{11} \text{ (Ya, Terlambat)} = \frac{100 \times 50}{200} = 25{,}000$
   - $E_{12} \text{ (Ya, Tepat Waktu)} = \frac{100 \times 150}{200} = 75{,}000$
   - $E_{21} \text{ (Tidak, Terlambat)} = \frac{100 \times 50}{200} = 25{,}000$
   - $E_{22} \text{ (Tidak, Tepat Waktu)} = \frac{100 \times 150}{200} = 75{,}000$

2. **Perhitungan Nilai Chi-Square ($\chi^2 = \sum \frac{(O - E)^2}{E}$)**:
   - Sel (Ya, Terlambat):
     $$\frac{(36 - 25{,}000)^2}{25{,}000} = \frac{(11)^2}{25{,}000} = \frac{121}{25{,}000} = 4{,}840$$
   - Sel (Ya, Tepat Waktu):
     $$\frac{(64 - 75{,}000)^2}{75{,}000} = \frac{(-11)^2}{75{,}000} = \frac{121}{75{,}000} \approx 1{,}613$$
   - Sel (Tidak, Terlambat):
     $$\frac{(14 - 25{,}000)^2}{25{,}000} = \frac{(-11)^2}{25{,}000} = \frac{121}{25{,}000} = 4{,}840$$
   - Sel (Tidak, Tepat Waktu):
     $$\frac{(86 - 75{,}000)^2}{75{,}000} = \frac{(11)^2}{75{,}000} = \frac{121}{75{,}000} \approx 1{,}613$$

   $$\chi^2 = 4{,}840 + 1{,}613 + 4{,}840 + 1{,}613 = 12{,}906 \quad (\text{atau } 12{,}907 \text{ dengan pembulatan eksak})$$

3. **Kesimpulan Uji Hipotesis**:
   - Derajat bebas ($df$) = $(2 - 1) \times (2 - 1) = 1$
   - Nilai kritis $\chi^2_{\text{tabel}} (df = 1, \alpha = 0{,}05) = 3{,}840$
   - Karena $\chi^2_{\text{hitung}} (12{,}907) > \chi^2_{\text{tabel}} (3{,}840)$, maka **Tolak $H_0$**.
   - **Artinya**: Terdapat hubungan/ketergantungan yang signifikan secara statistik antara fitur jenis layanan ekonomi dengan keterlambatan pengiriman. Fitur ini **sangat layak dipertahankan** dalam model.

---

### 3b. Desain Arsitektur ColumnTransformer, Fitur Baru, dan Penghapusan Kolom
1. **Rancangan Pipeline ColumnTransformer**:
   - **Fitur Numerik** (`berat_kg`, `jarak_km`):
     - `Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', RobustScaler() / StandardScaler())])`
     - Alasan: Menggunakan median imputer karena `jarak_km` miring (*skewed*) dan ada nilai kosong pada `berat_kg`.
   - **Fitur Kategorikal Nominal** (`kota_tujuan`, `jenis_layanan`):
     - `Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore'))])`
     - Alasan: Mengubah kategori tanpa tingkatan menjadi representasi biner (*dummy variables*).
   - **Fitur Kategorikal Ordinal** (`tingkat_prioritas`):
     - `Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('ordinal', OrdinalEncoder(categories=[['rendah', 'sedang', 'tinggi']]))])`
     - Alasan: Mempertahankan urutan hierarkis nilai (rendah = $0$, sedang = $1$, tinggi = $2$).

2. **Dua Usulan Fitur Baru (Feature Engineering)**:
   - **Fitur 1 (`kecepatan_tempuh_estimasi`)**: Rasio antara jarak pengiriman terhadap selisih waktu estimasi rute standar ($\frac{\text{jarak\_km}}{\text{estimasi\_jam}}$) untuk mengukur tingkat kesulitan logistik.
   - **Fitur 2 (`rasio_beban_jarak`)**: Rasio interaksi antara berat paket dan jarak tempuh ($\text{berat\_kg} \times \text{jarak\_km}$), yang merefleksikan konsumsi sumber daya kurir.

3. **Kolom yang Harus Dibuang Beserta Alasannya**:
   - **Kolom `tanggal_diterima`**: **Wajib dibuang**. Merupakan variabel target pasca-pengiriman (*post-event feature*). Memasukkannya memicu kebocoran data (**Data Leakage**) karena nilai tanggal terima belum tersedia saat prediksi dilakukan.
   - **Kolom `waktu_pemesanan` (raw timestamp murni)**: **Dibuang dalam bentuk mentahnya**. Format timestamp memiliki kardinalitas tak terhingga yang menyebabkan *overfitting*. Sebaiknya diekstrak terlebih dahulu menjadi fitur diskrit (misal: `jam_pemesanan`, `hari_dalam_minggu`) sebelum kolom aslinya dihapus.

---

## SOAL 4: DECISION TREE (CART, ID3 & PRUNING)

### 4a. Perhitungan Gini Impurity (CART)
Node Induk: $6$ Terlambat, $14$ Tepat Waktu (Total $N = 20$).

1. **Gini Node Induk**:
   $$p_1 = \frac{6}{20} = 0{,}300, \quad p_2 = \frac{14}{20} = 0{,}700$$
   $$\text{Gini}(\text{Induk}) = 1 - (p_1^2 + p_2^2) = 1 - (0{,}300^2 + 0{,}700^2) = 1 - (0{,}090 + 0{,}490) = 1 - 0{,}580 = 0{,}420$$

2. **Kandidat Split X (Cuaca Buruk di Rute?)**:
   - **Cabang Ya** ($4$ Terlambat, $1$ Tepat Waktu; $n_{\text{Ya}} = 5$):
     $$\text{Gini}(X_{\text{Ya}}) = 1 - \left[\left(\frac{4}{5}\right)^2 + \left(\frac{1}{5}\right)^2\right] = 1 - (0{,}640 + 0{,}040) = 1 - 0{,}680 = 0{,}320$$
   - **Cabang Tidak** ($2$ Terlambat, $13$ Tepat Waktu; $n_{\text{Tidak}} = 15$):
     $$\text{Gini}(X_{\text{Tidak}}) = 1 - \left[\left(\frac{2}{15}\right)^2 + \left(\frac{13}{15}\right)^2\right] = 1 - \left(\frac{4}{225} + \frac{169}{225}\right) = 1 - \frac{173}{225} = 1 - 0{,}769 = 0{,}231$$
   - **Gini Tertimbang Split X**:
     $$\text{Gini}(X) = \frac{5}{20} \times 0{,}320 + \frac{15}{20} \times 0{,}231 = 0{,}080 + 0{,}173 = 0{,}253$$
   - **Penurunan Gini ($\Delta \text{Gini}_X$)**:
     $$\Delta \text{Gini}_X = 0{,}420 - 0{,}253 = 0{,}167$$

3. **Kandidat Split Y (Layanan Ekonomi?)**:
   - **Cabang Ya** ($4$ Terlambat, $6$ Tepat Waktu; $n_{\text{Ya}} = 10$):
     $$\text{Gini}(Y_{\text{Ya}}) = 1 - \left[\left(\frac{4}{10}\right)^2 + \left(\frac{6}{10}\right)^2\right] = 1 - (0{,}160 + 0{,}360) = 1 - 0{,}520 = 0{,}480$$
   - **Cabang Tidak** ($2$ Terlambat, $8$ Tepat Waktu; $n_{\text{Tidak}} = 10$):
     $$\text{Gini}(Y_{\text{Tidak}}) = 1 - \left[\left(\frac{2}{10}\right)^2 + \left(\frac{8}{10}\right)^2\right] = 1 - (0{,}040 + 0{,}640) = 1 - 0{,}680 = 0{,}320$$
   - **Gini Tertimbang Split Y**:
     $$\text{Gini}(Y) = \frac{10}{20} \times 0{,}480 + \frac{10}{20} \times 0{,}320 = 0{,}240 + 0{,}160 = 0{,}400$$
   - **Penurunan Gini ($\Delta \text{Gini}_Y$)**:
     $$\Delta \text{Gini}_Y = 0{,}420 - 0{,}400 = 0{,}020$$

4. **Keputusan Algoritma CART**:
   - CART memilih kandidat split yang menghasilkan **Gini tertimbang terkecil** (atau reduksi impuritas terbesar).
   - Karena $\text{Gini}(X) = 0{,}253 < \text{Gini}(Y) = 0{,}400$, maka **CART memilih Split X**.

---

### 4b. Perhitungan Entropy dan Information Gain (ID3)
Rumus Entropy:
$$H(S) = - \sum p_i \log_2 (p_i) = - \sum p_i \frac{\ln(p_i)}{\ln(2)}$$

1. **Entropy Node Induk**:
   - $p_1 = 0{,}300 \implies \log_2(0{,}300) = \frac{\ln(0{,}300)}{\ln(2)} \approx -1{,}737$
   - $p_2 = 0{,}700 \implies \log_2(0{,}700) = \frac{\ln(0{,}700)}{\ln(2)} \approx -0{,}515$
   $$H(\text{Induk}) = - [0{,}300 \times (-1{,}737) + 0{,}700 \times (-0{,}515)] = - [-0{,}521 - 0{,}360] = 0{,}881 \text{ bit}$$

2. **Information Gain Split X**:
   - **Cabang Ya** ($4$ Terlambat, $1$ Tepat Waktu):
     $$p_1 = 0{,}800, \quad p_2 = 0{,}200$$
     $$H(X_{\text{Ya}}) = - [0{,}800 \log_2(0{,}800) + 0{,}200 \log_2(0{,}200)] = - [0{,}800(-0{,}322) + 0{,}200(-2{,}322)] = 0{,}722 \text{ bit}$$
   - **Cabang Tidak** ($2$ Terlambat, $13$ Tepat Waktu):
     $$p_1 = \frac{2}{15} \approx 0{,}133, \quad p_2 = \frac{13}{15} \approx 0{,}867$$
     $$H(X_{\text{Tidak}}) = - [0{,}133 \log_2(0{,}133) + 0{,}867 \log_2(0{,}867)] = - [0{,}133(-2{,}907) + 0{,}867(-0{,}206)] = 0{,}567 \text{ bit}$$
   - **Entropy Tertimbang Split X**:
     $$H(X) = \frac{5}{20} \times 0{,}722 + \frac{15}{20} \times 0{,}567 = 0{,}180 + 0{,}425 = 0{,}605 \text{ bit}$$
   - **Information Gain ($IG_X$)**:
     $$IG(X) = H(\text{Induk}) - H(X) = 0{,}881 - 0{,}605 = 0{,}276 \text{ bit}$$

3. **Information Gain Split Y**:
   - **Cabang Ya** ($4$ Terlambat, $6$ Tepat Waktu):
     $$p_1 = 0{,}400, \quad p_2 = 0{,}600$$
     $$H(Y_{\text{Ya}}) = - [0{,}400 \log_2(0{,}400) + 0{,}600 \log_2(0{,}600)] = - [0{,}400(-1{,}322) + 0{,}600(-0{,}737)] = 0{,}971 \text{ bit}$$
   - **Cabang Tidak** ($2$ Terlambat, $8$ Tepat Waktu):
     $$p_1 = 0{,}200, \quad p_2 = 0{,}800$$
     $$H(Y_{\text{Tidak}}) = - [0{,}200 \log_2(0{,}200) + 0{,}800 \log_2(0{,}800)] = - [0{,}200(-2{,}322) + 0{,}800(-0{,}322)] = 0{,}722 \text{ bit}$$
   - **Entropy Tertimbang Split Y**:
     $$H(Y) = \frac{10}{20} \times 0{,}971 + \frac{10}{20} \times 0{,}722 = 0{,}486 + 0{,}361 = 0{,}846 \text{ bit}$$
   - **Information Gain ($IG_Y$)**:
     $$IG(Y) = H(\text{Induk}) - H(Y) = 0{,}881 - 0{,}846 = 0{,}035 \text{ bit}$$

4. **Perbandingan dengan CART**:
   - ID3 memilih split dengan Information Gain tertinggi.
   - Karena $IG(X) = 0{,}276 \text{ bit} > IG(Y) = 0{,}035 \text{ bit}$, maka **ID3 memilih Split X**.
   - **Kesimpulan**: **Ya, algoritma ID3 memilih kandidat split yang sama persis dengan CART (yaitu Split X).**

---

### 4c. Cost-Complexity Pruning ($C_\alpha(T) = R(T) + \alpha \times |T|$)
Data Pohon:
- $T_1: R(T) = 0{,}110, \quad |T| = 22$ leaf
- $T_2: R(T) = 0{,}150, \quad |T| = 8$ leaf
- $T_3: R(T) = 0{,}230, \quad |T| = 3$ leaf

1. **Kondisi $\alpha = 0{,}004$**:
   - $C_{0{,}004}(T_1) = 0{,}110 + (0{,}004 \times 22) = 0{,}110 + 0{,}088 = 0{,}198$
   - $C_{0{,}004}(T_2) = 0{,}150 + (0{,}004 \times 8) = 0{,}150 + 0{,}032 = 0{,}182$
   - $C_{0{,}004}(T_3) = 0{,}230 + (0{,}004 \times 3) = 0{,}230 + 0{,}012 = 0{,}242$
   - **Pohon Terpilih**: **$T_2$** (karena memiliki nilai *cost-complexity* terendah, yaitu $0{,}182$).

2. **Kondisi $\alpha = 0{,}025$**:
   - $C_{0{,}025}(T_1) = 0{,}110 + (0{,}025 \times 22) = 0{,}110 + 0{,}550 = 0{,}660$
   - $C_{0{,}025}(T_2) = 0{,}150 + (0{,}025 \times 8) = 0{,}150 + 0{,}200 = 0{,}350$
   - $C_{0{,}025}(T_3) = 0{,}230 + (0{,}025 \times 3) = 0{,}230 + 0{,}075 = 0{,}305$
   - **Pohon Terpilih**: **$T_3$** (karena memiliki nilai *cost-complexity* terendah, yaitu $0{,}305$).

3. **Pengaruh Kenaikan Nilai $\alpha$ terhadap Bias dan Variance**:
   - Peningkatan parameter penalti $\alpha$ memberikan beban penalti yang semakin berat terhadap jumlah daun, sehingga algoritma memangkas cabang lebih agresif dan menghasilkan pohon yang lebih sederhana (*simpler tree*).
   - Akibatnya: **Bias meningkat** (kemungkinan underfitting naik) dan **Variance menurun** (model menjadi lebih stabil dan kebal terhadap derau pada data latih).

---

### 4d. Analisis Validation Curve `max_depth`
Tabel Kinerja:
| max_depth | 1 | 2 | 3 | 5 | 7 | 10 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Akurasi Train** | $0{,}690$ | $0{,}770$ | $0{,}840$ | $0{,}900$ | $0{,}950$ | $0{,}990$ |
| **Akurasi Validation** | $0{,}680$ | $0{,}750$ | **$0{,}810$** | $0{,}790$ | $0{,}760$ | $0{,}700$ |

1. **Nilai `max_depth` Terbaik**:
   - **Pilihan**: **`max_depth` = 3**.
   - **Alasan**: Menghasilkan akurasi pada set validasi tertinggi, yaitu **$0{,}810$ ($81{,}000\%$)**, dengan generalisasi yang sehat terhadap data train ($0{,}840$).

2. **Kondisi Model pada `max_depth` = 1**:
   - **Diagnosis**: **Underfitting** (Bias Tinggi).
   - Akurasi train ($0{,}690$) dan validasi ($0{,}680$) keduanya sangat rendah. Pohon hanya melakukan 1 kali pemisahan (*decision stump*), sehingga struktur model terlalu primitif untuk menangkap pola relasi fitur.

3. **Kondisi Model pada `max_depth` = 10**:
   - **Diagnosis**: **Overfitting** (Variance Tinggi).
   - Akurasi train sangat tinggi mendekati sempurna ($0{,}990$), namun akurasi validasi merosot tajam menjadi $0{,}700$ (terdapat *generalization gap* yang lebar sebesar $0{,}290$). Model terlalu dalam dan menghafal noise data latih.

---

## SOAL 5: K-NEAREST NEIGHBORS (KNN MANUAL & HIPERPARAMETER)

### 5a. Jarak Euclidean dari Titik Baru $q = (3, 5)$
Diketahui data latih:
- $D_1 = (4, 6)$ [Terlambat]
- $D_2 = (6, 9)$ [Tepat waktu]
- $D_3 = (0, 1)$ [Tepat waktu]
- $D_4 = (9, 13)$ [Terlambat]
- $D_5 = (3, 17)$ [Tepat waktu]

Rumus Jarak Euclidean:
$$d(q, D_i) = \sqrt{(x_{i, 1} - 3)^2 + (x_{i, 2} - 5)^2}$$

1. **Perhitungan Tiap Titik**:
   - **Jarak ke $D_1$**:
     $$d(q, D_1) = \sqrt{(4 - 3)^2 + (6 - 5)^2} = \sqrt{1^2 + 1^2} = \sqrt{2} \approx 1{,}414 \quad \text{[Terlambat]}$$
   - **Jarak ke $D_2$**:
     $$d(q, D_2) = \sqrt{(6 - 3)^2 + (9 - 5)^2} = \sqrt{3^2 + 4^2} = \sqrt{9 + 16} = \sqrt{25} = 5{,}000 \quad \text{[Tepat waktu]}$$
   - **Jarak ke $D_3$**:
     $$d(q, D_3) = \sqrt{(0 - 3)^2 + (1 - 5)^2} = \sqrt{(-3)^2 + (-4)^2} = \sqrt{9 + 16} = \sqrt{25} = 5{,}000 \quad \text{[Tepat waktu]}$$
   - **Jarak ke $D_4$**:
     $$d(q, D_4) = \sqrt{(9 - 3)^2 + (13 - 5)^2} = \sqrt{6^2 + 8^2} = \sqrt{36 + 64} = \sqrt{100} = 10{,}000 \quad \text{[Terlambat]}$$
   - **Jarak ke $D_5$**:
     $$d(q, D_5) = \sqrt{(3 - 3)^2 + (17 - 5)^2} = \sqrt{0^2 + 12^2} = \sqrt{144} = 12{,}000 \quad \text{[Tepat waktu]}$$

2. **Penentuan 3 Tetangga Terdekat ($k = 3$)**:
   - Peringkat 1: $D_1$ ($d = 1{,}414$) $\rightarrow$ Terlambat
   - Peringkat 2: $D_2$ ($d = 5{,}000$) $\rightarrow$ Tepat waktu
   - Peringkat 3: $D_3$ ($d = 5{,}000$) $\rightarrow$ Tepat waktu

3. **Prediksi Voting Mayoritas ($k = 3$)**:
   - Terlambat: $1$ suara ($D_1$)
   - Tepat Waktu: $2$ suara ($D_2, D_3$)
   - **Keputusan**: **Tepat Waktu** (karena mayoritas $2 > 1$).

4. **Prediksi Weighted KNN ($w = \frac{1}{d}$)**:
   - Bobot $D_1$ (Terlambat):
     $$w(D_1) = \frac{1}{1{,}414} \approx 0{,}707$$
   - Bobot $D_2$ (Tepat waktu):
     $$w(D_2) = \frac{1}{5{,}000} = 0{,}200$$
   - Bobot $D_3$ (Tepat waktu):
     $$w(D_3) = \frac{1}{5{,}000} = 0{,}200$$
   - **Total Bobot Kelas**:
     - $\text{Bobot Terlambat} = 0{,}707$
     - $\text{Bobot Tepat Waktu} = 0{,}200 + 0{,}200 = 0{,}400$
   - Karena $0{,}707 > 0{,}400$, maka **Keputusan Weighted KNN: Terlambat**.

5. **Apakah Keputusannya Berubah?**
   - **Ya, keputusannya berubah**. Pada voting mayoritas biasa prediksinya adalah **Tepat Waktu**, sedangkan pada weighted KNN berubah menjadi **Terlambat**. Hal ini terjadi karena titik $D_1$ letaknya sangat dekat ($1{,}414$) sehingga memperoleh bobot kedekatan yang jauh melampaui gabungan bobot dua titik lainnya yang berjarak lebih jauh ($5{,}000$).

---

### 5b. Jarak Manhattan ($L_1$) dari Titik Baru $q = (3, 5)$
Rumus Jarak Manhattan:
$$d_M(q, D_i) = |x_{i, 1} - 3| + |x_{i, 2} - 5|$$

1. **Perhitungan Tiap Titik**:
   - **Jarak ke $D_1$**: $|4 - 3| + |6 - 5| = 1 + 1 = 2{,}000 \quad \text{[Terlambat]}$
   - **Jarak ke $D_2$**: $|6 - 3| + |9 - 5| = 3 + 4 = 7{,}000 \quad \text{[Tepat waktu]}$
   - **Jarak ke $D_3$**: $|0 - 3| + |1 - 5| = 3 + 4 = 7{,}000 \quad \text{[Tepat waktu]}$
   - **Jarak ke $D_4$**: $|9 - 3| + |13 - 5| = 6 + 8 = 14{,}000 \quad \text{[Terlambat]}$
   - **Jarak ke $D_5$**: $|3 - 3| + |17 - 5| = 0 + 12 = 12{,}000 \quad \text{[Tepat waktu]}$

2. **Penentuan 3 Tetangga Terdekat ($k = 3$)**:
   - Peringkat 1: $D_1$ ($d_M = 2{,}000$) $\rightarrow$ Terlambat
   - Peringkat 2: $D_2$ ($d_M = 7{,}000$) $\rightarrow$ Tepat waktu
   - Peringkat 3: $D_3$ ($d_M = 7{,}000$) $\rightarrow$ Tepat waktu
   - **Kesimpulan Tetangga**: **Tiga tetangga terdekat tidak berubah**, tetap $D_1$, $D_2$, dan $D_3$.

3. **Evaluasi Prediksi**:
   - **Voting Mayoritas**: $1$ Terlambat vs $2$ Tepat Waktu $\implies$ **Tepat Waktu**.
   - **Weighted KNN ($w = \frac{1}{d_M}$)**:
     - $\text{Bobot Terlambat } (D_1) = \frac{1}{2{,}000} = 0{,}500$
     - $\text{Bobot Tepat Waktu } (D_2 + D_3) = \frac{1}{7{,}000} + \frac{1}{7{,}000} \approx 0{,}143 + 0{,}143 = 0{,}286$
     - Karena $0{,}500 > 0{,}286 \implies$ **Terlambat**.
   - **Kesimpulan Akhir**: Baik susunan 3 tetangga terdekat maupun hasil prediksinya **sama sekali tidak berubah** antara jarak Euclidean dan jarak Manhattan.

---

### 5c. Analisis 5-Fold Cross-Validation untuk Pemilihan $k$
Tabel Hasil:
| $k$ | 1 | 3 | 5 | 7 | 9 | 15 | 21 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Akurasi Train** | $1{,}000$ | $0{,}930$ | $0{,}900$ | $0{,}880$ | $0{,}860$ | $0{,}830$ | $0{,}800$ |
| **Rata-rata Akurasi CV** | $0{,}770$ | $0{,}820$ | $0{,}850$ | **$0{,}860$** | $0{,}840$ | $0{,}810$ | $0{,}780$ |

1. **Nilai $k$ Terbaik**:
   - **Pilihan**: **$k = 7$**.
   - **Alasan**: Menghasilkan rata-rata akurasi cross-validation tertinggi yaitu **$0{,}860$ ($86{,}000\%$)**.

2. **Kondisi Model pada $k = 1$**:
   - **Diagnosis**: **Overfitting Ekstrem** (Variance Sangat Tinggi).
   - Akurasi train bernilai sempurna ($1{,}000$ atau $100{,}000\%$) tetapi akurasi CV anjlok ke $0{,}770$. Batas keputusan (*decision boundary*) sangat berlekuk-lekuk dan sangat sensitif terhadap noise atau pencilan.

3. **Kondisi Model pada $k = 21$**:
   - **Diagnosis**: **Underfitting** (Bias Tinggi).
   - Akurasi train ($0{,}800$) dan CV ($0{,}780$) sama-sama rendah. Radius lingkungan pemungutan suara terlalu luas sehingga batas kelas menjadi terlalu halus (*oversmoothed*) dan mengaburkan pola lokal kelas minoritas.

4. **Mengapa Nilai $k$ Tidak Boleh Dipilih Berdasarkan Akurasi Data Train?**
   - Pada $k = 1$, setiap titik data latih selalu menjadi tetangga terdekat bagi dirinya sendiri (jarak $d = 0$), sehingga akurasi train akan selalu bernilai $100{,}000\%$. Akurasi train tidak mencerminkan kemampuan generalisasi model terhadap data baru. Oleh karena itu, optimasi hyperparameter $k$ wajib merujuk pada validasi silang (Cross-Validation).

---

### 5d. Urgensi Feature Scaling pada KNN dan Contoh Kasus
1. **Mengapa Fitur Wajib Diskalakan Sebelum Menghitung Jarak KNN?**
   - Algoritma KNN murni mengandalkan metrik jarak geometris (seperti Euclidean atau Manhattan). Jika fitur-fitur memiliki rentang nilai (*scale*) yang berbeda signifikan, maka fitur dengan rentang nilai besar secara matematis akan mendominasi perhitungan selisih kuadrat jarak, sedangkan fitur dengan skala kecil akan diabaikan pengaruhnya meskipun secara esensial lebih informatif.

2. **Contoh Kasus Nyata pada KirimCepat**:
   - Misalkan terdapat fitur `jarak_pengiriman` dengan rentang $5 - 65 \text{ km}$ ($5.000 - 65.000 \text{ meter}$) dan fitur `berat_paket` dengan rentang $1 - 18 \text{ kg}$.
   - Selisih jarak antar dua paket sebesar $10.000 \text{ meter}$ menghasilkan selisih kuadrat $(10.000)^2 = 100.000.000$.
   - Sedangkan selisih berat paket sebesar $5 \text{ kg}$ hanya menghasilkan selisih kuadrat $5^2 = 25$.
   - Akibatnya, nilai jarak $100.000.000$ menenggelamkan kontribusi fitur berat paket secara mutlak. Dengan standardisasi (misal: *StandardScaler* atau *MinMaxScaler*), kedua fitur akan berada pada rentang skala yang setara sehingga jarak dihitung secara adil dan akurat.
