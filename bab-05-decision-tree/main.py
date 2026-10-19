from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

csv_path = 'umkm_microloan_application.csv'
if not os.path.exists(csv_path):
    csv_path = r'd:\Anggit Arisa Siwi Pratama\Unindra\Machine Learning\ml-workspace\2-ml-playground\public\datasets\alternatif\umkm_microloan_application.csv'

if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)
    cols = [c for c in ['omset_bulanan', 'lama_usaha_tahun', 'skor_kredit', 'jumlah_pinjaman'] if c in df.columns]
    X = df[cols]
    y = df['status_layak'] if 'status_layak' in df.columns else df.iloc[:, -1]
    print(f"[OK] Dataset UMKM Microloan dimuat dari CSV: {X.shape[0]} baris.")
else:
    np.random.seed(42)
    n = 1000
    omset = np.random.uniform(5, 50, n)
    lama = np.random.uniform(1, 15, n)
    skor = np.random.randint(300, 850, n)
    pinjaman = np.random.uniform(10, 200, n)
    layak = ((omset > 15) & (skor > 600) & (lama >= 2)).astype(int)
    X = pd.DataFrame({'omset_bulanan': omset, 'lama_usaha_tahun': lama, 'skor_kredit': skor, 'jumlah_pinjaman': pinjaman})
    y = pd.Series(layak, name='status_layak')
    print(f"[OK] Dataset UMKM Microloan disintesis: {n} pengajuan kredit.")

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)
clf = DecisionTreeClassifier(criterion='gini', max_depth=3, ccp_alpha=0.01, random_state=42)
clf.fit(X_tr, y_tr)

importances = clf.feature_importances_
print(f"Akurasi Pohon Keputusan Uji : {clf.score(X_te, y_te)*100:.2f}%\n")
print("Aturan Keputusan Pohon Terpangkas:")
print(export_text(clf, feature_names=list(X.columns)))

# Visualisasi Feature Importance Horizontal Bar Chart
plt.style.use('dark_background')
fig, ax = plt.subplots(figsize=(6.5, 3.6), facecolor='#0B0E14')
ax.set_facecolor('#0B0E14')
sorted_idx = np.argsort(importances)
y_pos = np.arange(len(sorted_idx))
bars = ax.barh(y_pos, importances[sorted_idx], color='#34D399', edgecolor='#10B981', height=0.55)
ax.set_yticks(y_pos)
ax.set_yticklabels([list(X.columns)[i] for i in sorted_idx], color='#CBD5E1', fontsize=9)
ax.set_xlabel('Tingkat Kepentingan Fitur (Gini Importance)', color='#94A3B8', fontsize=9)
ax.set_title('UMKM Credit: Bobot Fitur Penentu Kelayakan Pinjaman', color='#F1F5F9', fontsize=10.5, pad=12, fontweight=600)
ax.tick_params(colors='#94A3B8', labelsize=8.5)
for bar in bars:
    val = bar.get_width()
    ax.text(val + 0.01, bar.get_y() + bar.get_height()/2, f'{val*100:.1f}%', va='center', color='#94A3B8', fontsize=8.5)
ax.set_xlim(0, max(importances) * 1.25)
for spine in ax.spines.values():
    spine.set_color('#1E293B')
plt.tight_layout()
os.makedirs('visual_output', exist_ok=True)
plt.savefig('visual_output/feature_importance_chart.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print("[OK] Grafik visual disimpan ke visual_output/feature_importance_chart.png")
