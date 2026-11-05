import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_digits

# 1. Pemuatan Dataset Digits & Mahasiswa Multidimensi
digits = load_digits()
X_raw = digits.data
y = digits.target
target_names = [str(i) for i in range(10)]

print(f'[OK] Dataset Digits dimuat: {X_raw.shape[0]} sampel, {X_raw.shape[1]} dimensi fitur.')

# 2. Standarisasi Fitur
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)

# 3. Principal Component Analysis (PCA)
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)
var_pc1 = pca.explained_variance_ratio_[0] * 100
var_pc2 = pca.explained_variance_ratio_[1] * 100
var_total = (pca.explained_variance_ratio_[0] + pca.explained_variance_ratio_[1]) * 100

# 4. Pembuktian Eigen Nilai 2x2 Manual (Soal Resmi Bab 11 Hal 66)
cov_sample = np.array([[2.0, 0.8], [0.8, 1.0]])
eigenvals, eigenvecs = np.linalg.eig(cov_sample)
lambda_1, lambda_2 = sorted(eigenvals, reverse=True)

# 5. t-SNE Manifold Non-Linier
tsne = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=800)
X_tsne = tsne.fit_transform(X_scaled[:600])
y_sub = y[:600]

print('\n' + '='*55)
print('HASIL EVALUASI REDUKSI DIMENSI PCA & t-SNE (BAB 11)')
print('='*55)
print(f'Pembuktian Eigen Bab 11 (Hal 66): Lambda 1 = {lambda_1:.4f}, Lambda 2 = {lambda_2:.4f}')
print(f'Explained Variance Ratio PC1    : {var_pc1:.2f}%')
print(f'Explained Variance Ratio PC2    : {var_pc2:.2f}%')
print(f'Total Variansi Terjelaskan (2D) : {var_total:.2f}%')
print(f'Reduksi Dimensi Fitur           : 64D -> 2D (Kompresi 96.88%)')
print('='*55)

os.makedirs('visual_output', exist_ok=True)
plt.style.use('dark_background')

# Gambar 1: Proyeksi Linier PCA 2D
fig, ax = plt.subplots(figsize=(6.5, 4.5), facecolor='#0B0E14')
ax.set_facecolor('#0B0E14')
scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='tab10', alpha=0.6, s=20)
cbar = fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label('Label Digit (0-9)', color='#94A3B8', fontsize=8.5)
ax.set_title(f'Bab 11: Proyeksi 2D PCA (Variansi Terjelaskan {var_total:.1f}%)', color='#F1F5F9', fontsize=10.5, pad=12, fontweight=600)
ax.set_xlabel(f'Principal Component 1 ({var_pc1:.1f}%)', color='#94A3B8', fontsize=9)
ax.set_ylabel(f'Principal Component 2 ({var_pc2:.1f}%)', color='#94A3B8', fontsize=9)
ax.grid(True, linestyle='--', alpha=0.2)
plt.tight_layout()
plt.savefig('visual_output/11_pca_2d_projection.png', dpi=300, facecolor=fig.get_facecolor(), bbox_inches='tight')
plt.close()

# Gambar 2: Manifold Embedding t-SNE Non-Linier
fig, ax = plt.subplots(figsize=(6.5, 4.5), facecolor='#0B0E14')
ax.set_facecolor('#0B0E14')
scatter = ax.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_sub, cmap='tab10', alpha=0.7, s=22)
cbar = fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label('Label Digit (0-9)', color='#94A3B8', fontsize=8.5)
ax.set_title('Bab 11: Manifold Non-Linier t-SNE (Perplexity=30)', color='#F1F5F9', fontsize=10.5, pad=12, fontweight=600)
ax.set_xlabel('Dimensi t-SNE 1', color='#94A3B8', fontsize=9)
ax.set_ylabel('Dimensi t-SNE 2', color='#94A3B8', fontsize=9)
ax.grid(True, linestyle='--', alpha=0.2)
plt.tight_layout()
plt.savefig('visual_output/11_tsne_manifold_projection.png', dpi=300, facecolor=fig.get_facecolor(), bbox_inches='tight')
plt.close()

print('[OK] Visual output Bab 11 berhasil disimpan di visual_output/')

