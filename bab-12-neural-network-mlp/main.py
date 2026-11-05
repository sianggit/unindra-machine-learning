import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. Pembuktian Forward Pass Satu Neuron (Soal Resmi Bab 12 Hal 72)
# x = [0.5, -0.2, 0.8], w = [0.4, 0.6, -0.3], b = 0.1
x_manual = np.array([0.5, -0.2, 0.8])
w_manual = np.array([0.4, 0.6, -0.3])
b_manual = 0.1
z_manual = np.dot(x_manual, w_manual) + b_manual
a_relu = max(0.0, z_manual)

# 2. Pemuatan & Pelatihan Multilayer Perceptron (MLP)
digits = load_digits()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(digits.data)
X_tr, X_te, y_tr, y_te = train_test_split(X_scaled, digits.target, test_size=0.25, random_state=42, stratify=digits.target)

mlp = MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu', solver='adam', max_iter=250, random_state=42)
mlp.fit(X_tr, y_tr)
y_pred = mlp.predict(X_te)
acc = accuracy_score(y_te, y_pred)
loss_history = mlp.loss_curve_

print('\n' + '='*55)
print('HASIL EVALUASI MULTILAYER PERCEPTRON (BAB 12)')
print('='*55)
print(f'Forward Pass Satu Neuron (Hal 72):')
print(f'- Input x : {x_manual.tolist()} | Bobot w: {w_manual.tolist()}')
print(f'- Nilai Linier z = (w . x + b) : {z_manual:.4f}')
print(f'- Aktivasi Non-Linier ReLU(z)   : {a_relu:.4f}')
print(f'Arsitektur Jaringan MLP         : Input (64) -> Dense(64) -> Dense(32) -> Output(10)')
print(f'Akurasi Uji (Test Accuracy)     : {acc*100:.2f}%')
print(f'Nilai Loss Akhir (Cross-Entropy): {mlp.loss_:.4f}')
print(f'Jumlah Iterasi Sampai Konvergen : {mlp.n_iter_} iterasi')
print('='*55)

os.makedirs('visual_output', exist_ok=True)
plt.style.use('dark_background')

# Gambar 1: Kurva Konvergensi Loss Pelatihan
fig, ax = plt.subplots(figsize=(6.5, 4.2), facecolor='#0B0E14')
ax.set_facecolor('#0B0E14')
ax.plot(range(1, len(loss_history) + 1), loss_history, color='#38BDF8', linewidth=2, label='Training Loss')
ax.set_title(f'Bab 12: Kurva Konvergensi Loss MLP (Akurasi {acc*100:.1f}%)', color='#F1F5F9', fontsize=10.5, pad=12, fontweight=600)
ax.set_xlabel('Epoch (Iterasi)', color='#94A3B8', fontsize=9)
ax.set_ylabel('Loss (Cross-Entropy)', color='#94A3B8', fontsize=9)
ax.legend(facecolor='#1E293B', edgecolor='#334155', fontsize=8.5)
ax.grid(True, linestyle='--', alpha=0.25)
plt.tight_layout()
plt.savefig('visual_output/12_mlp_neural_network_loss_curve.png', dpi=300, facecolor=fig.get_facecolor(), bbox_inches='tight')
plt.close()

# Gambar 2: Heatmap Confusion Matrix Multikelas
cm = confusion_matrix(y_te, y_pred)
fig, ax = plt.subplots(figsize=(6.0, 4.5), facecolor='#0B0E14')
ax.set_facecolor('#0B0E14')
cax = ax.imshow(cm, cmap='Blues')
fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
ax.set_xticks(range(10))
ax.set_yticks(range(10))
ax.set_xticklabels(range(10), color='#94A3B8', fontsize=8.5)
ax.set_yticklabels(range(10), color='#94A3B8', fontsize=8.5)
for i in range(10):
    for j in range(10):
        val = cm[i, j]
        if val > 0:
            ax.text(j, i, str(val), ha='center', va='center', color='white' if val > 20 else '#CBD5E1', fontsize=8)
ax.set_title('Bab 12: Confusion Matrix Prediksi Digit MLP', color='#F1F5F9', fontsize=10.5, pad=12, fontweight=600)
ax.set_xlabel('Label Prediksi', color='#94A3B8', fontsize=9)
ax.set_ylabel('Label Aktual', color='#94A3B8', fontsize=9)
plt.tight_layout()
plt.savefig('visual_output/12_mlp_confusion_matrix.png', dpi=300, facecolor=fig.get_facecolor(), bbox_inches='tight')
plt.close()

print('[OK] Visual output Bab 12 berhasil disimpan di visual_output/')

