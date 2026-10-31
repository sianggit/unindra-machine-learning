import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# 1. Pemuatan Dataset Solar Panel Faults
csv_path = 'solar_panel_faults.csv'
if not os.path.exists(csv_path):
    csv_path = r'd:\Anggit Arisa Siwi Pratama\Unindra\Machine Learning\ml-workspace\2-ml-playground\public\datasets\alternatif\solar_panel_faults.csv'

if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)
    X = df[['temperature_c', 'voltage_v']]
    y = df['is_fault']
    print(f'[OK] Dataset Solar Panel dimuat: {len(df)} titik pantau modul surya.')
else:
    np.random.seed(99)
    n = 400
    tmp = np.random.uniform(25, 75, n)
    vlt = np.random.uniform(20, 45, n)
    cur = np.random.uniform(2, 12, n)
    fault = (((tmp > 55) & (vlt < 30)) | (cur < 4)).astype(int)
    X = pd.DataFrame({'temperature_c': tmp, 'voltage_v': vlt})
    y = pd.Series(fault, name='is_fault')
    print(f'[OK] Dataset disintesis secara deterministik: {n} sampel.')

# 2. Skalasi Standar & Pembagian Partisi
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_tr, X_te, y_tr, y_te = train_test_split(X_scaled, y, test_size=0.25, stratify=y, random_state=42)

# 3. Model Linear SVM & Kalkulasi Margin Teoritis
svm_linear = SVC(kernel='linear', C=1.0, random_state=42)
svm_linear.fit(X_tr, y_tr)
w = svm_linear.coef_[0]
b = svm_linear.intercept_[0]
w_norm = np.linalg.norm(w)
margin = 2.0 / w_norm

# 4. Benchmarking Kernel SVM (Linear, RBF, Polynomial)
kernels = ['linear', 'rbf', 'poly']
kernel_accs = {}
for k in kernels:
    model = SVC(kernel=k, C=1.0, random_state=42)
    model.fit(X_tr, y_tr)
    kernel_accs[k] = accuracy_score(y_te, model.predict(X_te))

print('\n' + '='*50)
print('HASIL EVALUASI SUPPORT VECTOR MACHINE (BAB 08)')
print('='*50)
print(f'Koefisien Bobot (w)  : [{w[0]:.4f}, {w[1]:.4f}]')
print(f'Bias (b)             : {b:.4f}')
print(f'Norma Vektor ||w||   : {w_norm:.4f}')
print(f'Lebar Margin (2/||w||): {margin:.4f}')
print(f'Jumlah Support Vector: {len(svm_linear.support_)} titik')
for k, acc in kernel_accs.items():
    print(f'Akurasi Kernel {k.upper():6s}: {acc*100:.2f}%')
print('='*50)

os.makedirs('visual_output', exist_ok=True)
plt.style.use('dark_background')

# 5. Visualisasi Hyperplane dan Margin Batas Keputusan
fig, ax = plt.subplots(figsize=(6.2, 4.5), facecolor='#0B0E14')
ax.set_facecolor('#0B0E14')

# Scatter sampel data
scatter = ax.scatter(X_tr[:, 0], X_tr[:, 1], c=y_tr, cmap='coolwarm', s=30, alpha=0.75, edgecolors='none')
# Highlight support vectors
sv = svm_linear.support_vectors_
ax.scatter(sv[:, 0], sv[:, 1], s=80, facecolors='none', edgecolors='#FBBF24', lw=1.5, label='Support Vectors')

# Garis Hyperplane & Margin
x_points = np.linspace(X_tr[:, 0].min() - 0.5, X_tr[:, 0].max() + 0.5, 200)
y_hyperplane = -(w[0] * x_points + b) / w[1]
y_margin_pos = -(w[0] * x_points + b - 1) / w[1]
y_margin_neg = -(w[0] * x_points + b + 1) / w[1]

ax.plot(x_points, y_hyperplane, color='#38BDF8', lw=2, label='Hyperplane (w^T x + b = 0)')
ax.plot(x_points, y_margin_pos, color='#94A3B8', linestyle='--', lw=1.2, label=f'Margin +1 / -1 (M={margin:.2f})')
ax.plot(x_points, y_margin_neg, color='#94A3B8', linestyle='--', lw=1.2)

ax.set_xlim([X_tr[:, 0].min() - 0.5, X_tr[:, 0].max() + 0.5])
ax.set_ylim([X_tr[:, 1].min() - 0.5, X_tr[:, 1].max() + 0.5])
ax.set_xlabel('Temperatur Modul (Z-Score)', color='#94A3B8', fontsize=9)
ax.set_ylabel('Tegangan Output (Z-Score)', color='#94A3B8', fontsize=9)
ax.set_title('Bab 08: Maximal Margin Hyperplane SVM Linear (Panel Surya)', color='#F1F5F9', fontsize=10.5, pad=12, fontweight=600)
ax.legend(loc='lower left', facecolor='#1E293B', edgecolor='#334155', fontsize=8)
ax.tick_params(colors='#94A3B8', labelsize=8.5)
for spine in ax.spines.values():
    spine.set_color('#1E293B')
plt.tight_layout()
plt.savefig('visual_output/08_svm_hyperplane_margins.png', dpi=300)
plt.close()

# 6. Visualisasi Komparasi Akurasi Antar Kernel
fig, ax = plt.subplots(figsize=(5.5, 3.8), facecolor='#0B0E14')
ax.set_facecolor('#0B0E14')
names = ['Linear', 'RBF (Gaussian)', 'Polynomial']
acc_vals = [kernel_accs['linear'], kernel_accs['rbf'], kernel_accs['poly']]
bars = ax.bar(names, [v * 100 for v in acc_vals], color=['#38BDF8', '#34D399', '#A78BFA'], edgecolor='#0F172A', width=0.45)
ax.set_ylabel('Akurasi Uji (%)', color='#94A3B8', fontsize=9)
ax.set_title('Bab 08: Komparasi Kinerja Fungsi Kernel SVM', color='#F1F5F9', fontsize=10.5, pad=12, fontweight=600)
ax.set_ylim(0, 110)
ax.tick_params(colors='#94A3B8', labelsize=8.5)
for bar in bars:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, h + 2, f'{h:.2f}%', ha='center', color='#F1F5F9', fontsize=8.5, weight='bold')
for spine in ax.spines.values():
    spine.set_color('#1E293B')
plt.tight_layout()
plt.savefig('visual_output/08_svm_kernel_comparison.png', dpi=300)
plt.close()
print('[OK] Visual output Bab 08 berhasil disimpan di visual_output/')

