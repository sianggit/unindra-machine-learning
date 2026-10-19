# === SIMULASI KOMPUTASI NUMERIK: KISI-KISI UTS SOAL 1 S.D. 5 (KIRIMCEPAT) ===
import numpy as np
import pandas as pd
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

print("================================================================")
print("  LEMBAR JAWABAN KOMPUTASI KISI-KISI UTS MACHINE LEARNING       ")
print("  Mahasiswa : Anggit Arisa Siwi Pratama (NPM: 202343570045)     ")
print("================================================================")

# SOAL 1: Studi Kasus KirimCepat & Baseline
N_total = 10000
n_tepat = 8800
n_lambat = 1200
acc_baseline = n_tepat / N_total
print(f"\n[SOAL 1] Akurasi Baseline Menebak Mayoritas: {acc_baseline*100:.2f}%")
print("         Recall Kelas Positif (Terlambat) : 0.00% (Accuracy Paradox)")

# SOAL 2: Outlier IQR & SMOTE
berat = [1, 2, 2, 3, 3, 4, 5, 18]
q1, q3 = np.percentile(berat, 25), np.percentile(berat, 75)
iqr = q3 - q1
upper_bound = q3 + 1.5 * iqr
print(f"\n[SOAL 2] Berat Paket: Q1={q1:.2f}, Q3={q3:.2f}, IQR={iqr:.2f}, Batas Atas={upper_bound:.2f} kg")
print(f"         Outlier terdeteksi: {[x for x in berat if x > upper_bound]} kg")

# Sintesis SMOTE
xi = np.array([2.0, 7.0])
xj = np.array([8.0, 4.0])
lam = 1.0 / 3.0
x_new = xi + lam * (xj - xi)
print(f"         Sintesis Titik SMOTE: xi={xi}, xj={xj} -> x_new={x_new}")

# SOAL 3: Uji Chi-Square Kontingensi 2x2
O = np.array([[36, 64], [14, 86]])
row_totals = O.sum(axis=1)
col_totals = O.sum(axis=0)
total = O.sum()
E = np.outer(row_totals, col_totals) / total
chi2 = np.sum((O - E)**2 / E)
print(f"\n[SOAL 3] Nilai Chi-Square Hitung = {chi2:.3f} (Kritis df=1, a=0.05: 3.841)")
print(f"         Keputusan: Chi2 Hitung ({chi2:.3f}) > 3.841 -> TOLAK H0 (Signifikan)")

# SOAL 4: Seleksi Split CART vs ID3
gini_induk = 1.0 - ((6/20)**2 + (14/20)**2)
gini_x = (8/20)*(1 - ((5/8)**2 + (3/8)**2)) + (12/20)*(1 - ((1/12)**2 + (11/12)**2))
print(f"\n[SOAL 4] Gini Induk: {gini_induk:.3f} | Gini Split X (Cuaca): {gini_x:.3f}")
print("         CART & ID3 konsisten memilih Split X sebagai pemisah terbaik.")

# SOAL 5: Jarak Manual K-NN
q = np.array([3.0, 5.0])
D = np.array([[1.0, 2.0], [2.0, 5.0], [4.0, 4.0], [5.0, 8.0], [6.0, 3.0]])
dist_euc = np.linalg.norm(D - q, axis=1)
print(f"\n[SOAL 5] Jarak Euclidean q(3,5) ke D1-D5: {np.round(dist_euc, 3)}")

# Visualisasi Plot Komposit UTS
plt.style.use('dark_background')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.5, 3.6), facecolor='#0B0E14')
for ax in (ax1, ax2):
    ax.set_facecolor('#0B0E14')
    for spine in ax.spines.values():
        spine.set_color('#1E293B')
    ax.tick_params(colors='#94A3B8', labelsize=8.5)

# Plot 1: SMOTE
ax1.scatter([2, 8], [7, 4], color='#38BDF8', s=80, label='Sampel Asli (xi, xj)', zorder=3)
ax1.scatter([x_new[0]], [x_new[1]], color='#34D399', s=120, marker='*', label='Titik SMOTE (x_new)', zorder=4)
ax1.plot([2, 8], [7, 4], color='#64748B', linestyle='--', linewidth=1.2)
ax1.set_title('Soal 2: Interpolasi Geometris SMOTE', color='#F1F5F9', fontsize=9.5, fontweight=600)
ax1.legend(loc='upper right', facecolor='#0B0E14', edgecolor='#1E293B', fontsize=8)

# Plot 2: Chi-Square
ax2.bar(['Chi2 Hitung', 'Kritis Tabel'], [chi2, 3.841], color=['#34D399', '#64748B'], width=0.45)
ax2.axhline(3.841, color='#FBBF24', linestyle=':', label='Ambang Kritis 3.841')
ax2.set_title('Soal 3: Uji Signifikansi Chi-Square', color='#F1F5F9', fontsize=9.5, fontweight=600)
ax2.set_ylabel('Nilai Statistik', color='#94A3B8', fontsize=8.5)
ax2.legend(loc='upper right', facecolor='#0B0E14', edgecolor='#1E293B', fontsize=8)

plt.tight_layout()
os.makedirs('visual_output', exist_ok=True)
plt.savefig('visual_output/uts_kirimcepat_summary.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print("[OK] Grafik visual UTS disimpan ke visual_output/uts_kirimcepat_summary.png")
