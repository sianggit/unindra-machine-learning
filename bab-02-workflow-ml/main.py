from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 1. Pemuatan Data Ketenagakerjaan
csv_path = 'hr_attrition_dataset.csv'
if not os.path.exists(csv_path):
    csv_path = r'd:\Anggit Arisa Siwi Pratama\Unindra\Machine Learning\ml-workspace\2-ml-playground\public\datasets\alternatif\hr_attrition_dataset.csv'

if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)
    if 'left' in df.columns:
        cols = [c for c in ['satisfaction_level', 'last_evaluation', 'number_project', 'average_montly_hours', 'time_spend_company'] if c in df.columns]
        X = df[cols].dropna()
        y = df.loc[X.index, 'left'].astype(int)
    elif 'Attrition' in df.columns:
        cols = [c for c in ['Age', 'DailyRate', 'DistanceFromHome', 'MonthlyIncome', 'TotalWorkingYears'] if c in df.columns]
        X = df[cols].dropna()
        y = (df.loc[X.index, 'Attrition'] == 'Yes').astype(int)
    else:
        cols = df.select_dtypes(include=[np.number]).columns[:-1]
        X = df[cols].dropna()
        y = df.iloc[X.index, -1].astype(int)
    print(f"[OK] Dataset IBM HR Attrition dimuat dari CSV: {X.shape[0]} baris.")
else:
    np.random.seed(42)
    n = 1470
    X = pd.DataFrame({
        'satisfaction_level': np.random.uniform(0.1, 1.0, n),
        'last_evaluation': np.random.uniform(0.3, 1.0, n),
        'number_project': np.random.randint(2, 7, n),
        'average_montly_hours': np.random.randint(96, 310, n),
        'time_spend_company': np.random.randint(2, 10, n)
    })
    y = np.random.choice([0, 1], size=n, p=[0.84, 0.16])
    print(f"[OK] Dataset IBM HR Attrition disintesis: {n} baris (16% Attrition).")

# 2. Standardisasi Fitur & Komparasi K-Fold
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = LogisticRegression(max_iter=1000, random_state=42)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
rkf = KFold(n_splits=5, shuffle=True, random_state=42)

scores_strat = cross_val_score(model, X_scaled, y, cv=skf, scoring='accuracy')
scores_rand = cross_val_score(model, X_scaled, y, cv=rkf, scoring='accuracy')

print(f"Stratified 5-Fold Akurasi : {[round(float(s)*100, 2) for s in scores_strat]}%")
print(f"Rata-rata Stratified       : {scores_strat.mean()*100:.2f}% (Std: {scores_strat.std():.4f})")
print(f"Rata-rata Random K-Fold   : {scores_rand.mean()*100:.2f}% (Std: {scores_rand.std():.4f})")

# 3. Visualisasi Stabilitas per Fold
plt.style.use('dark_background')
fig, ax = plt.subplots(figsize=(6.5, 3.6), facecolor='#0B0E14')
ax.set_facecolor('#0B0E14')
folds = [f'Fold {i+1}' for i in range(5)]
x = np.arange(5)
width = 0.35
ax.bar(x - width/2, scores_rand*100, width, label='Random K-Fold', color='#64748B')
ax.bar(x + width/2, scores_strat*100, width, label='Stratified K-Fold', color='#34D399')
ax.axhline(scores_strat.mean()*100, color='#FBBF24', linestyle='--', linewidth=1.2, label=f'Mean Stratified ({scores_strat.mean()*100:.1f}%)')
ax.set_ylabel('Akurasi (%)', color='#94A3B8', fontsize=9)
ax.set_title('IBM HR Attrition: Stabilitas Akurasi per Fold (Random vs Stratified)', color='#F1F5F9', fontsize=10.5, pad=12, fontweight=600)
ax.set_xticks(x)
ax.set_xticklabels(folds, color='#94A3B8', fontsize=9)
ax.tick_params(colors='#94A3B8', labelsize=8.5)
ax.set_ylim(70, 100)
ax.legend(loc='lower right', facecolor='#0B0E14', edgecolor='#1E293B', fontsize=8.5)
for spine in ax.spines.values():
    spine.set_color('#1E293B')
plt.tight_layout()
os.makedirs('visual_output', exist_ok=True)
plt.savefig('visual_output/cv_stability_barchart.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print("[OK] Grafik visual disimpan ke visual_output/cv_stability_barchart.png")
