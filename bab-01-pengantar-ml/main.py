import numpy as np
import pandas as pd
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 1. Pemuatan Data & Representasi Matriks Fitur
csv_path = 'penguins_morphology.csv'
if not os.path.exists(csv_path):
    csv_path = r'd:\Anggit Arisa Siwi Pratama\Unindra\Machine Learning\ml-workspace\2-ml-playground\public\datasets\alternatif\penguins_morphology.csv'

if os.path.exists(csv_path):
    data = pd.read_csv(csv_path)
    fitur_cols = [c for c in ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g', 'culmen_length_mm', 'culmen_depth_mm'] if c in data.columns]
    features = data[fitur_cols].dropna().values
    print(f"[OK] Dataset Palmer Penguins dimuat dari CSV: {data.shape[0]} baris, {len(fitur_cols)} fitur.")
else:
    np.random.seed(42)
    features = np.column_stack([
        np.random.normal(43.9, 5.4, 344),
        np.random.normal(17.1, 1.9, 344),
        np.random.normal(200.9, 14.0, 344),
        np.random.normal(4201, 801, 344)
    ])
    print("[OK] Dataset Palmer Penguins disintesis secara deterministik: 344 spesimen.")

# 2. Vektorisasi Statistik Tanpa Loop & Standardisasi Z-Score
mean_vec = np.mean(features, axis=0)
std_vec = np.std(features, axis=0)
z_scores = (features - mean_vec) / std_vec

# 3. Matriks Kovariansi Ruang Fitur
cov_matrix = np.cov(z_scores, rowvar=False)
print("\nMatriks Kovariansi Fitur Morfometrik Terstandarisasi:")
print(np.round(cov_matrix, 4))
print(f"\n[Selesai] Dimensi Matriks Fitur: {features.shape[0]} baris x {features.shape[1]} kolom.")

# 4. Visualisasi Heatmap Kovariansi
plt.style.use('dark_background')
fig, ax = plt.subplots(figsize=(6.5, 3.8), facecolor='#0B0E14')
ax.set_facecolor('#0B0E14')
cax = ax.imshow(cov_matrix, cmap='coolwarm', vmin=-1, vmax=1)
fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
cols_label = ['Bill Length', 'Bill Depth', 'Flipper Len', 'Body Mass'][:features.shape[1]]
ax.set_xticks(range(len(cols_label)))
ax.set_yticks(range(len(cols_label)))
ax.set_xticklabels(cols_label, fontsize=8.5, color='#94A3B8', rotation=15)
ax.set_yticklabels(cols_label, fontsize=8.5, color='#94A3B8')
for i in range(len(cols_label)):
    for j in range(len(cols_label)):
        val = cov_matrix[i, j]
        ax.text(j, i, f'{val:.2f}', ha='center', va='center', color='white' if abs(val) > 0.5 else '#CBD5E1', fontsize=9, fontweight=600)
ax.set_title('Palmer Penguins: Matriks Kovariansi Fitur Morfometrik', color='#F1F5F9', fontsize=10.5, pad=12, fontweight=600)
for spine in ax.spines.values():
    spine.set_color('#1E293B')
plt.tight_layout()
os.makedirs('visual_output', exist_ok=True)
plt.savefig('visual_output/covariance_matrix_heatmap.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print("[OK] Grafik visual disimpan ke visual_output/covariance_matrix_heatmap.png")
