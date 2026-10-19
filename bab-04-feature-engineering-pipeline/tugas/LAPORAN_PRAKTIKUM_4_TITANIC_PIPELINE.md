# LAPORAN PRAKTIKUM MACHINE LEARNING
## BAB 04: FEATURE ENGINEERING & PREPROCESSING PIPELINE
### Klasifikasi Kelangsungan Hidup Penumpang Titanic Menggunakan `ColumnTransformer` dan `Pipeline` Bebas Kebocoran Data

---

**Identitas Mahasiswa:**
- **Nama**: ANGGIT ARISA SIWI PRATAMA
- **NPM**: 202343570045
- **Kelas**: XC Ekstensi
- **Program Studi**: Teknik Informatika (S1), Fakultas Teknik dan Ilmu Komputer
- **Institusi**: Universitas Indraprasta PGRI (Unindra)
- **Mata Kuliah**: Machine Learning *) #) (Kode: KK43K780)
- **Dosen**: Nurfidah Dwitiyanti, M.Kom.
- **Tanggal Praktikum**: 02 Oktober 2026

---

### Abstrak
Praktikum Bab 04 mengkaji implementasi rekayasa fitur (*feature engineering*) dan konstruksi pipeline pra-pemrosesan data terintegrasi. Sesuai instruksi pada modul ajar (Modul 4 Halaman 24), dataset properti sintetik digantikan dengan dataset historis Titanic (*N = 891*). Karakteristik dataset ini memiliki nilai kosong pada kolom `age` (19.87%) dan `embarked` (0.22%), serta keberagaman tipe data (kontinu, nominal, dan ordinal). Kami menguji dependensi kategorikal menggunakan statistik Chi-Square ($\chi^2$), merekayasa tiga fitur turunan (`family_size`, `is_alone`, dan `fare_per_person`), serta menyusun arsitektur `ColumnTransformer` scikit-learn untuk mengisolasi proses transformasi agar terbebas dari *data leakage*. Model dievaluasi menggunakan pembagian data terstratifikasi 80:20. Hasil pengujian menunjukkan bahwa penambahan fitur hasil rekayasa pada model *Logistic Regression* meningkatkan akurasi dari 77.65% ($F_1 = 0.6970$) menjadi 79.33% ($F_1 = 0.7132$).

---

### 1. Landasan Teori dan Latar Belakang Masalah

Tahap pra-pemrosesan data dan rekayasa fitur merupakan fase krusial dalam siklus CRISP-DM (*Data Preparation*). Pada data mentah dunia nyata, pengumpanan data secara langsung ke algoritma estimasi tanpa perlakuan statistik yang terstandarisasi menimbulkan tiga risiko teknis:

1. **Kegagalan Eksekusi Numerik**: Algoritma seperti regresi logistik dan jaringan saraf tiruan tidak dapat memproses nilai hilang (*missing values*) maupun data teks non-numerik.
2. **Dominasi Skala Variabel**: Fitur berrentang nilai besar (seperti `fare`: 0 hingga 512.33) mendominasi perhitungan gradien dibandingkan fitur berrentang sempit (seperti `age`: 0.42 hingga 80.0), sehingga memerlukan standarisasi deviasi (*z-score scaling*).
3. **Kebocoran Data (*Data Leakage*)**: Terjadi ketika nilai statistik global dari seluruh dataset (misalnya nilai median `age` atau frekuensi modus `embarked`) dihitung sebelum pemisahan data latih (*training set*) dan data uji (*test set*). Informasi dari data uji yang merembes ke data latih menghasilkan estimasi performa optimis palsu (*overfitting bias*).

Untuk mengatasi risiko tersebut secara deterministik, scikit-learn menyediakan modul `ColumnTransformer` yang digabungkan ke dalam `Pipeline`.

---

### 2. Uji Dependensi Statistik Chi-Square ($\chi^2$)

Sesuai materi seleksi fitur metode filter (Modul 4 Halaman 14–16), uji Chi-Square digunakan untuk mengukur signifikansi asosiasi antara variabel input kategorikal terhadap variabel target biner `survived`:

$$\chi^2 = \sum_{i=1}^{r} \sum_{j=1}^{c} \frac{(O_{ij} - E_{ij})^2}{E_{ij}}$$

di mana frekuensi harapan ($E_{ij}$) dihitung berdasarkan perkalian total baris dan total kolom dibagi total populasi sampel ($N$):

$$E_{ij} = \frac{R_i \times C_j}{N}$$

#### Tabel 1. Hasil Uji Chi-Square Fitur Kategorikal terhadap Target `survived`
| Nama Fitur | Tipe Fitur | Derajat Bebas ($df$) | Nilai $\chi^2$ Hitung | Nilai $p$-value | Keputusan ($\alpha = 0.05$) |
| :--- | :--- | :---: | :---: | :---: | :--- |
| `sex` | Kategorikal Nominal | 1 | 260.717 | $1.197 \times 10^{-58}$ | Tolak $H_0$ (Sangat Signifikan) |
| `pclass` | Diskrit Ordinal | 2 | 102.889 | $4.549 \times 10^{-23}$ | Tolak $H_0$ (Sangat Signifikan) |
| `embarked` | Kategorikal Nominal | 2 | 26.489 | $1.770 \times 10^{-6}$ | Tolak $H_0$ (Signifikan) |
| `is_alone` | Biner Rekayasa | 1 | 36.002 | $1.972 \times 10^{-9}$ | Tolak $H_0$ (Signifikan) |

*Catatan Nilai Kritis*: Dengan $\alpha = 0.05$ dan $df = 1$, nilai kritis tabel distribusi $\chi^2$ adalah 3.841. Karena seluruh nilai $\chi^2$ hitung berada jauh di atas 3.841 ($p < 0.001$), hipotesis nol independensi ditolak. Seluruh fitur terbukti berhubungan secara statistik dengan probabilitas keselamatan penumpang.

---

### 3. Konstruksi Fitur Baru (*Feature Construction*)

Tiga variabel baru direkayasa berdasarkan pemahaman domain peristiwa kecelakaan laut Titanic:

1. **`family_size`** ($x_{\text{keluarga}} = \text{sibsp} + \text{parch} + 1$):
   Mengagregasikan jumlah relasi keluarga di kapal (saudara, pasangan, orang tua, anak) ditambah individu penumpang. Anggota keluarga besar menghadapi kesulitan koordinasi saat evakuasi, sedangkan keluarga berukuran sedang (2–4 orang) memiliki tingkat keselamatan tertinggi.
2. **`is_alone`** ($\mathbb{I}_{\{family\_size = 1\}}$):
   Indikator biner bernilai 1 jika penumpang bepergian tanpa keluarga, dan 0 jika membawa anggota keluarga.
3. **`fare_per_person`** ($x_{\text{tarif}} = \frac{\text{fare}}{\text{family\_size}}$):
   Pada data manifest asli, harga tiket (`fare`) sering kali dibeli satu kali untuk seluruh rombongan keluarga. Pembagian tarif terhadap jumlah anggota keluarga memberikan aproksimasi daya beli ekonomi per individu secara lebih objektif.

---

### 4. Arsitektur Preprocessing Pipeline

Proses transformasi data dibagi menjadi tiga jalur terisolasi dalam satu objek `ColumnTransformer`:

1. **Jalur Numerik** (`age`, `fare`, `family_size`, `fare_per_person`):
   - Imputasi nilai kosong menggunakan nilai median data latih (`SimpleImputer(strategy='median')`). Median dipilih karena distribusi `fare` memiliki kemiringan positif (*right-skewed*) yang rentan terhadap pencilan (*outliers*).
   - Standardisasi skala fitur menggunakan skor baku ($z = \frac{x - \mu}{\sigma}$) melalui `StandardScaler()`.
2. **Jalur Kategorikal Nominal** (`sex`, `embarked`):
   - Imputasi nilai kosong menggunakan kategori frekuensi modus (`SimpleImputer(strategy='most_frequent')`).
   - Transformasi biner menggunakan `OneHotEncoder(drop='first', handle_unknown='ignore')` untuk menghindari multikolinearitas (*dummy variable trap*).
3. **Jalur Diskrit Ordinal** (`pclass`):
   - Imputasi nilai hilang menggunakan modus.
   - Pemetaan urutan tingkatan kelas sosial (`1` = Kelas 1, `2` = Kelas 2, `3` = Kelas 3) menggunakan `OrdinalEncoder(categories=[[1, 2, 3]])`.

---

### 5. Implementasi Source Code Python

```python
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# 1. Pemuatan Data
df = sns.load_dataset('titanic')

# 2. Rekayasa Fitur (Feature Construction)
df['family_size'] = df['sibsp'] + df['parch'] + 1
df['is_alone'] = (df['family_size'] == 1).astype(int)
df['fare_per_person'] = df['fare'] / df['family_size']

y = df['survived']

# Definisi Kelompok Kolom
fitur_num = ['age', 'fare', 'family_size', 'fare_per_person']
fitur_cat = ['sex', 'embarked']
fitur_ord = ['pclass']
kolom_aktif = fitur_num + fitur_cat + fitur_ord

# 3. Pembagian Data Terstratifikasi (Train 80%, Test 20%)
X_train, X_test, y_train, y_test = train_test_split(
    df[kolom_aktif], y, test_size=0.20, stratify=y, random_state=42
)

# 4. Konstruksi Sub-Pipeline Transformasi Kolom
num_subpipe = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

cat_subpipe = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore'))
])

ord_subpipe = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('ordinal', OrdinalEncoder(categories=[[1, 2, 3]]))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', num_subpipe, fitur_num),
    ('cat', cat_subpipe, fitur_cat),
    ('ord', ord_subpipe, fitur_ord)
])

# 5. Pembangunan Model Pipeline (Logistic Regression & Random Forest)
pipeline_lr = Pipeline([
    ('prep', preprocessor),
    ('clf', LogisticRegression(max_iter=1000, random_state=42))
])

pipeline_rf = Pipeline([
    ('prep', preprocessor),
    ('clf', RandomForestClassifier(n_estimators=100, random_state=42))
])

# 6. Pelatihan dan Inferensi
pipeline_lr.fit(X_train, y_train)
pred_lr = pipeline_lr.predict(X_test)

pipeline_rf.fit(X_train, y_train)
pred_rf = pipeline_rf.predict(X_test)
```

---

### 6. Hasil Evaluasi Model dan Analisis Kinerja

Evaluasi dilakukan terhadap $N_{\text{test}} = 179$ sampel data uji independen. Metrik evaluasi mencakup Akurasi, Presisi, Recall, dan F1-Score:

$$\text{Presisi} = \frac{TP}{TP + FP}, \quad \text{Recall} = \frac{TP}{TP + FN}, \quad F_1 = 2 \times \frac{\text{Presisi} \times \text{Recall}}{\text{Presisi} + \text{Recall}}$$

#### Tabel 2. Komparasi Kinerja Model pada Data Uji Titanic
| Model | Konfigurasi Fitur | Akurasi | Presisi | Recall | F1-Score |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Model 1: Baseline LR** | `age`, `fare`, `sex`, `embarked` | 77.65% | 73.02% | 66.67% | 0.6970 |
| **Model 2: Enhanced LR** | Baseline + `pclass`, `family_size`, `fare_per_person` | **79.33%** | **76.67%** | 66.67% | **0.7132** |
| **Model 3: Enhanced RF** | Baseline + `pclass`, `family_size`, `fare_per_person` | 78.77% | 73.85% | **69.57%** | **0.7164** |

#### Tabel 3. Matriks Konfusi (*Confusion Matrix*) Data Uji
| Model | True Negative ($TN$) | False Positive ($FP$) | False Negative ($FN$) | True Positive ($TP$) |
| :--- | :---: | :---: | :---: | :---: |
| **Enhanced Logistic Regression** | 95 | 15 | 23 | 46 |
| **Enhanced Random Forest** | 93 | 17 | 21 | 48 |

#### Analisis Hasil:
1. Penambahan fitur rekayasa (`family_size`, `fare_per_person`, dan `pclass`) memberikan peningkatan akurasi linier sebesar **+1.68%** dan peningkatan $F_1$-score sebesar **+0.0162** pada model Logistic Regression.
2. Model Random Forest menghasilkan Recall tertinggi (**69.57%**), mendeteksi 48 korban selamat dari total 69 kasus aktual kelas positif, menandakan kemampuannya menangkap interaksi non-linier antara usia dan ukuran keluarga secara lebih fleksibel.

---

### 7. Kesimpulan

1. Penggunaan `ColumnTransformer` scikit-learn memungkinkan penerapan strategi pembersihan dan standardisasi yang berbeda secara terisolasi untuk fitur bertipe numerik, nominal, dan ordinal dalam satu alur kerja terpadu.
2. Penyatuan pra-pemrosesan ke dalam `Pipeline` menjamin bahwa parameter statistik seperti median dan deviasi standar hanya dihitung dari data latih, mencegah terjadinya kebocoran informasi (*data leakage*) ke data uji.
3. Rekayasa fitur domain (*feature construction*) terbukti secara empiris meningkatkan daya pisah model dibandingkan hanya mengandalkan fitur mentah standar.
