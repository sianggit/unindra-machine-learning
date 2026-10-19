from sklearn.impute import SimpleImputer
from sklearn.utils import resample
import numpy as np
import pandas as pd
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 1. Pemuatan Data Transaksi Finansial
csv_path = 'financial_fraud_data.csv'
if not os.path.exists(csv_path):
    csv_path = r'd:\Anggit Arisa Siwi Pratama\Unindra\Machine Learning\ml-workspace\2-ml-playground\public\datasets\alternatif\financial_fraud_data.csv'

if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)
    if 'is_fraud' in df.columns:
        fitur = [c for c in ['amount', 'dist_home_km', 'dist_last_tx_km', 'ratio_to_median'] if c in df.columns]
        X = df[fitur]
        y = df['is_fraud'].astype(int)
    elif 'isFraud' in df.columns:
        fitur = [c for c in ['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest'] if c in df.columns]
        X = df[fitur]
        y = df['isFraud'].astype(int)
    else:
        fitur = df.columns[:-1]
        X = df[fitur]
        y = df.iloc[:, -1].astype(int)
    print(f"[OK] Dataset Financial Fraud dimuat dari CSV: {X.shape[0]} baris.")
else:
    np.random.seed(42)
    n = 2000
    X = pd.DataFrame(np.random.exponential(scale=50000, size=(n, 5)),
                     columns=['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest'])
    mask = np.random.rand(*X.shape) < 0.05
    X[mask] = np.nan
    y = np.random.choice([0, 1], size=n, p=[0.98, 0.02])
    print(f"[OK] Dataset Financial Fraud disintesis: {n} transaksi (2% Fraud).")

# 2. Imputasi Robust Median
imp = SimpleImputer(strategy='median')
X_imp = imp.fit_transform(X)

# 3. Resampling Penyeimbangan Kelas Minoritas (SMOTE Simulative Resampling)
X_df = pd.DataFrame(X_imp, columns=X.columns)
target_series = pd.Series(y, name='target').reset_index(drop=True)
df_all = pd.concat([X_df, target_series], axis=1)

df_maj = df_all[df_all['target'] == 0]
df_min = df_all[df_all['target'] == 1]
df_min_upsampled = resample(df_min, replace=True, n_samples=len(df_maj), random_state=42)
df_resampled = pd.concat([df_maj, df_min_upsampled])

n_maj_before, n_min_before = int(sum(y == 0)), int(sum(y == 1))
n_maj_after, n_min_after = int(sum(df_resampled['target'] == 0)), int(sum(df_resampled['target'] == 1))

print("[Preprocessing & Penyeimbangan Kelas Selesai]")
print(f"- Missing Value Pasca-Imputasi       : {np.isnan(X_imp).sum()}")
print(f"- Distribusi Kelas Sebelum Resampling: Normal={n_maj_before}, Fraud={n_min_before}")
print(f"- Distribusi Kelas Pasca-Resampling  : Normal={n_maj_after}, Fraud={n_min_after}")

# 4. Visualisasi Distribusi Kelas Sebelum vs Sesudah
plt.style.use('dark_background')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.6), facecolor='#0B0E14')
for ax in (ax1, ax2):
    ax.set_facecolor('#0B0E14')
    for spine in ax.spines.values():
        spine.set_color('#1E293B')
    ax.tick_params(colors='#94A3B8', labelsize=8.5)

bars1 = ax1.bar(['Normal (0)', 'Fraud (1)'], [n_maj_before, n_min_before], color=['#38BDF8', '#F43F5E'])
ax1.set_title(f'Sebelum Resampling\n(Rasio Ketimpangan {n_maj_before}:{n_min_before})', color='#F1F5F9', fontsize=9.5, fontweight=600)
ax1.set_ylabel('Jumlah Transaksi', color='#94A3B8', fontsize=9)
for bar in bars1:
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 15, f'{int(bar.get_height())}', ha='center', color='#E2E8F0', fontsize=8.5)

bars2 = ax2.bar(['Normal (0)', 'Fraud (1)'], [n_maj_after, n_min_after], color=['#38BDF8', '#34D399'])
ax2.set_title('Sesudah Resampling SMOTE\n(Distribusi Seimbang 1:1)', color='#F1F5F9', fontsize=9.5, fontweight=600)
for bar in bars2:
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 15, f'{int(bar.get_height())}', ha='center', color='#E2E8F0', fontsize=8.5)

plt.tight_layout()
os.makedirs('visual_output', exist_ok=True)
plt.savefig('visual_output/smote_class_distribution.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print("[OK] Grafik visual disimpan ke visual_output/smote_class_distribution.png")
