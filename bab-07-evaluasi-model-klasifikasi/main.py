import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc

# 1. Pemuatan Dataset Cyber Intrusion
csv_path = 'cyber_intrusion_logs.csv'
if not os.path.exists(csv_path):
    csv_path = r'd:\Anggit Arisa Siwi Pratama\Unindra\Machine Learning\ml-workspace\2-ml-playground\public\datasets\alternatif\cyber_intrusion_logs.csv'

if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)
    X = df[['packet_rate_per_sec', 'byte_error_rate', 'failed_login_attempts', 'session_duration_sec']]
    y = df['is_intrusion']
    print(f'[OK] Dataset Cyber Intrusion dimuat: {len(df)} transaksi jaringan.')
else:
    np.random.seed(88)
    n = 1000
    pkt = np.random.exponential(15, n) + np.random.choice([0, 80], n, p=[0.8, 0.2])
    err = np.random.beta(0.5, 5, n)
    flg = np.random.poisson(0.5, n)
    sec = np.random.exponential(30, n)
    y = ((pkt > 50) | (flg >= 3) | (err > 0.4)).astype(int)
    X = pd.DataFrame({'packet_rate_per_sec': pkt, 'byte_error_rate': err, 'failed_login_attempts': flg, 'session_duration_sec': sec})
    y = pd.Series(y, name='is_intrusion')
    print(f'[OK] Dataset disintesis secara deterministik: {n} sampel.')

# 2. Stratified Partitioning & Training Model
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)
clf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
clf.fit(X_tr, y_tr)
y_pred = clf.predict(X_te)
y_prob = clf.predict_proba(X_te)[:, 1]

# 3. Kalkulasi Matriks Diagnostik Evaluasi
cm = confusion_matrix(y_te, y_pred)
tn, fp, fn, tp = cm.ravel()
acc = accuracy_score(y_te, y_pred)
prec = precision_score(y_te, y_pred)
rec = recall_score(y_te, y_pred)
f1 = f1_score(y_te, y_pred)
fpr, tpr, _ = roc_curve(y_te, y_prob)
roc_auc = auc(fpr, tpr)

# Stratified K-Fold Cross Validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(clf, X, y, cv=skf, scoring='f1')

print('\n' + '='*50)
print('HASIL EVALUASI MODEL KLASIFIKASI (BAB 07)')
print('='*50)
print(f'True Positive (TP)  : {tp:4d}  | False Positive (FP) : {fp:4d}')
print(f'False Negative (FN) : {fn:4d}  | True Negative (TN)  : {tn:4d}')
print(f'Akurasi Uji         : {acc*100:.2f}%')
print(f'Presisi (Precision) : {prec*100:.2f}%')
print(f'Recall (Sensitivitas): {rec*100:.2f}%')
print(f'F1-Score Harmonik   : {f1*100:.2f}%')
print(f'ROC-AUC Score       : {roc_auc:.4f}')
print(f'Stratified 5-Fold F1: {cv_scores.mean()*100:.2f}% (+/- {cv_scores.std()*100:.2f}%)')
print('='*50)

os.makedirs('visual_output', exist_ok=True)
plt.style.use('dark_background')

# 4. Visualisasi Heatmap Confusion Matrix
fig, ax = plt.subplots(figsize=(5.5, 4.2), facecolor='#0B0E14')
ax.set_facecolor('#0B0E14')
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax,
            annot_kws={'size': 13, 'weight': 'bold', 'color': '#FFFFFF'},
            xticklabels=['Normal (0)', 'Intrusion (1)'],
            yticklabels=['Normal (0)', 'Intrusion (1)'])
ax.set_title(f'Bab 07: Confusion Matrix Heatmap (F1={f1*100:.1f}%)', color='#F1F5F9', fontsize=10.5, pad=12, fontweight=600)
ax.set_xlabel('Label Prediksi Model', color='#94A3B8', fontsize=9)
ax.set_ylabel('Label Aktual Lapangan', color='#94A3B8', fontsize=9)
ax.tick_params(colors='#94A3B8', labelsize=8.5)
plt.tight_layout()
plt.savefig('visual_output/07_confusion_matrix_heatmap.png', dpi=300)
plt.close()

# 5. Visualisasi ROC Curve & AUC
fig, ax = plt.subplots(figsize=(5.5, 4.2), facecolor='#0B0E14')
ax.set_facecolor('#0B0E14')
ax.plot(fpr, tpr, color='#38BDF8', lw=2.2, label=f'Random Forest (AUC = {roc_auc:.4f})')
ax.plot([0, 1], [0, 1], color='#64748B', lw=1.2, linestyle='--', label='Random Chance (AUC = 0.50)')
ax.set_xlim([-0.02, 1.02])
ax.set_ylim([-0.02, 1.05])
ax.set_xlabel('False Positive Rate (FPR)', color='#94A3B8', fontsize=9)
ax.set_ylabel('True Positive Rate (TPR / Recall)', color='#94A3B8', fontsize=9)
ax.set_title('Bab 07: Kurva ROC-AUC Deteksi Serangan Siber', color='#F1F5F9', fontsize=10.5, pad=12, fontweight=600)
ax.legend(loc='lower right', facecolor='#1E293B', edgecolor='#334155', fontsize=8.5)
ax.grid(color='#1E293B', linestyle=':', alpha=0.6)
ax.tick_params(colors='#94A3B8', labelsize=8.5)
for spine in ax.spines.values():
    spine.set_color('#1E293B')
plt.tight_layout()
plt.savefig('visual_output/07_roc_auc_curve.png', dpi=300)
plt.close()
print('[OK] Visual output Bab 07 berhasil disimpan di visual_output/')
