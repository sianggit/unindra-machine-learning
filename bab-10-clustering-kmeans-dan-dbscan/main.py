import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.preprocessing import StandardScaler

# 1. Pemuatan Dataset Spotify Audio Tracks
csv_path = 'spotify_audio_tracks.csv'
if not os.path.exists(csv_path):
    csv_path = r'd:\Anggit Arisa Siwi Pratama\Unindra\Machine Learning\ml-workspace\2-ml-playground\public\datasets\alternatif\spotify_audio_tracks.csv'

if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)
    feature_cols = [c for c in ['danceability', 'energy', 'valence', 'tempo', 'acousticness', 'loudness'] if c in df.columns]
    X_raw = df[feature_cols].dropna().values
    print(f'[OK] Dataset Spotify Audio Tracks dimuat: {len(X_raw)} lagu, {len(feature_cols)} fitur audio.')
else:
    np.random.seed(42)
    n = 600
    danceability = np.random.uniform(0.3, 0.9, n)
    energy = np.random.uniform(0.2, 0.95, n)
    valence = np.random.uniform(0.1, 0.9, n)
    tempo = np.random.normal(120, 20, n)
    acousticness = np.random.beta(1.5, 3.0, n)
    X_raw = np.column_stack([danceability, energy, valence, tempo, acousticness])
    feature_cols = ['danceability', 'energy', 'valence', 'tempo', 'acousticness']
    print(f'[OK] Dataset disintesis secara deterministik: {n} lagu.')

# 2. Standarisasi Fitur
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)

# 3. Analisis Elbow Method & Silhouette Score (k = 2..8)
k_range = range(2, 9)
inertias = []
silhouettes = []

for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    inertias.append(km.inertia_)
    sil = silhouette_score(X_scaled, km.labels_)
    silhouettes.append(sil)

best_k = list(k_range)[np.argmax(silhouettes)]
best_km = KMeans(n_clusters=best_k, random_state=42, n_init=10)
best_km.fit(X_scaled)
ch_score = calinski_harabasz_score(X_scaled, best_km.labels_)

# 4. Clustering Berbasis Kepadatan DBSCAN
dbscan = DBSCAN(eps=1.2, min_samples=6)
db_labels = dbscan.fit_predict(X_scaled)
n_clusters_db = len(set(db_labels)) - (1 if -1 in db_labels else 0)
n_noise_db = list(db_labels).count(-1)

print('\n' + '='*55)
print('HASIL EVALUASI CLUSTERING K-MEANS & DBSCAN (BAB 10)')
print('='*55)
print(f'Jumlah Klaster Optimal K-Means : k = {best_k}')
print(f'Silhouette Score Tertinggi     : {max(silhouettes):.4f}')
print(f'Calinski-Harabasz Index        : {ch_score:.2f}')
print(f'DBSCAN Jumlah Klaster Terbentuk: {n_clusters_db} klaster')
print(f'DBSCAN Titik Noise (Outlier)   : {n_noise_db} titik ({n_noise_db/len(X_scaled)*100:.1f}%)')
print('='*55)

# 5. Visualisasi Output Grafik 300 DPI
os.makedirs('visual_output', exist_ok=True)
plt.style.use('dark_background')

# Gambar 1: Elbow Curve & Silhouette
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2), facecolor='#0B0E14')
ax1.set_facecolor('#0B0E14')
ax2.set_facecolor('#0B0E14')

ax1.plot(k_range, inertias, 'o-', color='#38BDF8', linewidth=2, markersize=6)
ax1.set_title('Elbow Method: Inersia vs Jumlah Klaster', color='#F1F5F9', fontsize=10.5, pad=10, fontweight=600)
ax1.set_xlabel('Nilai k (Jumlah Klaster)', color='#94A3B8', fontsize=9)
ax1.set_ylabel('Inersia (Within-Cluster Sum of Squares)', color='#94A3B8', fontsize=9)
ax1.grid(True, linestyle='--', alpha=0.3)

ax2.plot(k_range, silhouettes, 's-', color='#34D399', linewidth=2, markersize=6)
ax2.axvline(best_k, color='#F87171', linestyle='--', label=f'Optimal k={best_k}')
ax2.set_title(f'Silhouette Analysis (Optimal k={best_k})', color='#F1F5F9', fontsize=10.5, pad=10, fontweight=600)
ax2.set_xlabel('Nilai k (Jumlah Klaster)', color='#94A3B8', fontsize=9)
ax2.set_ylabel('Rata-rata Koefisien Silhouette', color='#94A3B8', fontsize=9)
ax2.legend(loc='upper right', facecolor='#1E293B', edgecolor='#334155', fontsize=8.5)
ax2.grid(True, linestyle='--', alpha=0.3)

plt.tight_layout()
plt.savefig('visual_output/10_kmeans_elbow_silhouette.png', dpi=300, facecolor=fig.get_facecolor(), bbox_inches='tight')
plt.close()

# Gambar 2: Scatter Plot Sebaran Klaster Audio
fig, ax = plt.subplots(figsize=(6.5, 4.5), facecolor='#0B0E14')
ax.set_facecolor('#0B0E14')
scatter = ax.scatter(X_scaled[:, 0], X_scaled[:, 1], c=best_km.labels_, cmap='viridis', alpha=0.75, s=35, edgecolors='none')
centers = best_km.cluster_centers_
ax.scatter(centers[:, 0], centers[:, 1], c='#EF4444', marker='X', s=140, linewidths=2, edgecolors='#FFFFFF', label='Centroid')
ax.set_title(f'Bab 10: Partisi Klaster Musik K-Means (k={best_k})', color='#F1F5F9', fontsize=10.5, pad=12, fontweight=600)
ax.set_xlabel(f'Fitur Terstandarisasi: {feature_cols[0]}', color='#94A3B8', fontsize=9)
ax.set_ylabel(f'Fitur Terstandarisasi: {feature_cols[1]}', color='#94A3B8', fontsize=9)
ax.legend(facecolor='#1E293B', edgecolor='#334155', fontsize=8.5)
ax.grid(True, linestyle='--', alpha=0.2)

plt.tight_layout()
plt.savefig('visual_output/10_dbscan_clustering_scatter.png', dpi=300, facecolor=fig.get_facecolor(), bbox_inches='tight')
plt.close()

print('[OK] Visual output Bab 10 berhasil disimpan di visual_output/')

