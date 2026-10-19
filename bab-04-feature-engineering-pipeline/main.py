from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import numpy as np
import pandas as pd
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

csv_path = 'titanic.csv'
if not os.path.exists(csv_path):
    csv_path = r'd:\Anggit Arisa Siwi Pratama\Unindra\Machine Learning\ml-workspace\2-ml-playground\public\datasets\titanic.csv'

if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)
    df.columns = [c.lower() for c in df.columns]
    print(f"[OK] Dataset Titanic dimuat dari CSV: {df.shape[0]} baris.")
else:
    np.random.seed(42)
    n = 891
    df = pd.DataFrame({
        'pclass': np.random.choice([1, 2, 3], size=n, p=[0.24, 0.21, 0.55]),
        'sex': np.random.choice(['male', 'female'], size=n, p=[0.65, 0.35]),
        'age': np.where(np.random.rand(n) < 0.2, np.nan, np.random.normal(30, 14, n)),
        'sibsp': np.random.choice([0, 1, 2], size=n, p=[0.7, 0.2, 0.1]),
        'parch': np.random.choice([0, 1], size=n, p=[0.8, 0.2]),
        'fare': np.random.exponential(32, n),
        'embarked': np.random.choice(['S', 'C', 'Q'], size=n, p=[0.7, 0.2, 0.1]),
        'survived': np.random.choice([0, 1], size=n, p=[0.62, 0.38])
    })
    print(f"[OK] Dataset Titanic disintesis: {n} baris.")

# 1. Rekayasa Fitur Domain
df['family_size'] = df['sibsp'] + df['parch'] + 1
df['is_alone'] = (df['family_size'] == 1).astype(int)
df['fare_per_person'] = df['fare'] / df['family_size']

fitur_num = ['age', 'fare', 'family_size', 'fare_per_person']
fitur_cat = ['sex', 'embarked']
fitur_ord = ['pclass']
X = df[fitur_num + fitur_cat + fitur_ord]
y = df['survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=42)

# 2. Pipeline ColumnTransformer Bebas Data Leakage
preprocessor = ColumnTransformer(transformers=[
    ('num', Pipeline([('imp', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), fitur_num),
    ('cat', Pipeline([('imp', SimpleImputer(strategy='most_frequent')), ('ohe', OneHotEncoder(drop='first', handle_unknown='ignore'))]), fitur_cat),
    ('ord', Pipeline([('imp', SimpleImputer(strategy='most_frequent')), ('ord', OrdinalEncoder(categories=[[1, 2, 3]]))]), fitur_ord)
])

pipe_lr = Pipeline([('prep', preprocessor), ('clf', LogisticRegression(max_iter=1000, random_state=42))])
pipe_rf = Pipeline([('prep', preprocessor), ('clf', RandomForestClassifier(n_estimators=100, random_state=42))])

pipe_lr.fit(X_train, y_train)
pipe_rf.fit(X_train, y_train)

preds_rf = pipe_rf.predict(X_test)
acc_lr = accuracy_score(y_test, pipe_lr.predict(X_test))
acc_rf = accuracy_score(y_test, preds_rf)

print("[Hasil Evaluasi Pipeline Bebas Data Leakage (Test N=179)]")
print(f"1. Logistic Regression Akurasi : {acc_lr*100:.2f}% | F1: {f1_score(y_test, pipe_lr.predict(X_test)):.4f}")
print(f"2. Random Forest Akurasi       : {acc_rf*100:.2f}% | F1: {f1_score(y_test, preds_rf):.4f}")

# 3. Visualisasi Confusion Matrix Random Forest
cm = confusion_matrix(y_test, preds_rf)
plt.style.use('dark_background')
fig, ax = plt.subplots(figsize=(5.5, 3.8), facecolor='#0B0E14')
ax.set_facecolor('#0B0E14')
cax = ax.imshow(cm, cmap='Blues', vmin=0)
fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
labels = ['Meninggal (0)', 'Selamat (1)']
ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(labels, color='#94A3B8', fontsize=9)
ax.set_yticklabels(labels, color='#94A3B8', fontsize=9)
ax.set_xlabel('Prediksi Model', color='#94A3B8', fontsize=9)
ax.set_ylabel('Aktual Data', color='#94A3B8', fontsize=9)
for i in range(2):
    for j in range(2):
        ax.text(j, i, f'{cm[i, j]}', ha='center', va='center', color='white' if cm[i, j] > cm.max()/2 else '#CBD5E1', fontsize=12, fontweight=700)
ax.set_title(f'Titanic Pipeline: Confusion Matrix (Akurasi: {acc_rf*100:.1f}%)', color='#F1F5F9', fontsize=10.5, pad=12, fontweight=600)
for spine in ax.spines.values():
    spine.set_color('#1E293B')
plt.tight_layout()
os.makedirs('visual_output', exist_ok=True)
plt.savefig('visual_output/confusion_matrix_heatmap.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print("[OK] Grafik visual disimpan ke visual_output/confusion_matrix_heatmap.png")
