import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score

# 1. Pemuatan Dataset Telco Churn
csv_path = 'telco_churn_customers.csv'
if not os.path.exists(csv_path):
    csv_path = r'd:\Anggit Arisa Siwi Pratama\Unindra\Machine Learning\ml-workspace\2-ml-playground\public\datasets\alternatif\telco_churn_customers.csv'

if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)
    feature_cols = ['tenure_months', 'monthly_charges', 'support_calls', 'contract_monthly', 'internet_fiber']
    X = df[feature_cols]
    y = df['churn']
    print(f'[OK] Dataset Telco Churn dimuat: {len(df)} pelanggan.')
else:
    np.random.seed(77)
    n = 600
    tenure = np.random.randint(1, 72, n)
    charge = np.random.uniform(50000, 850000, n)
    calls = np.random.poisson(1.5, n)
    contract = np.random.choice([0, 1], n, p=[0.4, 0.6])
    fiber = np.random.choice([0, 1], n, p=[0.5, 0.5])
    churn = (((tenure < 12) & (charge > 300000)) | (calls >= 3) | (contract == 1)).astype(int)
    X = pd.DataFrame({'tenure_months': tenure, 'monthly_charges': charge, 'support_calls': calls, 'contract_monthly': contract, 'internet_fiber': fiber})
    y = pd.Series(churn, name='churn')
    print(f'[OK] Dataset disintesis secara deterministik: {n} sampel.')

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)

# 2. Benchmarking Algoritma Ensemble
models = {
    'Single Tree (Baseline)': DecisionTreeClassifier(max_depth=4, random_state=42),
    'Random Forest (Bagging)': RandomForestClassifier(n_estimators=100, oob_score=True, random_state=42),
    'AdaBoost (Boosting)': AdaBoostClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
}

results = {}
for name, m in models.items():
    m.fit(X_tr, y_tr)
    preds = m.predict(X_te)
    results[name] = {
        'accuracy': accuracy_score(y_te, preds),
        'f1': f1_score(y_te, preds)
    }

rf_model = models['Random Forest (Bagging)']
oob_score = rf_model.oob_score_
importances = rf_model.feature_importances_

print('\n' + '='*50)
print('HASIL EVALUASI ENSEMBLE LEARNING (BAB 09)')
print('='*50)
for name, metrics in results.items():
    print(f'{name:24s} | Akurasi: {metrics["accuracy"]*100:6.2f}% | F1: {metrics["f1"]*100:6.2f}%')
print(f'Out-of-Bag (OOB) Score RF: {oob_score*100:.2f}% (Estimasi error OOB: {(1-oob_score)*100:.2f}%)')
print('='*50)

os.makedirs('visual_output', exist_ok=True)
plt.style.use('dark_background')

# 3. Visualisasi Perbandingan Model Ensemble
fig, ax = plt.subplots(figsize=(6.5, 4.0), facecolor='#0B0E14')
ax.set_facecolor('#0B0E14')
names = list(results.keys())
f1_vals = [results[m]['f1'] * 100 for m in names]
bars = ax.barh(names, f1_vals, color=['#64748B', '#38BDF8', '#34D399', '#A78BFA'], height=0.5, edgecolor='#0F172A')
ax.set_xlabel('F1-Score Uji (%)', color='#94A3B8', fontsize=9)
ax.set_title('Bab 09: Komparasi Kinerja Algoritma Ensemble (Telco Churn)', color='#F1F5F9', fontsize=10.5, pad=12, fontweight=600)
ax.set_xlim(0, 115)
ax.tick_params(colors='#94A3B8', labelsize=8.5)
for bar in bars:
    w = bar.get_width()
    ax.text(w + 1.5, bar.get_y() + bar.get_height()/2, f'{w:.2f}%', va='center', color='#F1F5F9', fontsize=8.5, weight='bold')
for spine in ax.spines.values():
    spine.set_color('#1E293B')
plt.tight_layout()
plt.savefig('visual_output/09_ensemble_methods_comparison.png', dpi=300)
plt.close()

# 4. Visualisasi Feature Importance Random Forest
fig, ax = plt.subplots(figsize=(6.5, 3.8), facecolor='#0B0E14')
ax.set_facecolor('#0B0E14')
sorted_idx = np.argsort(importances)
y_pos = np.arange(len(sorted_idx))
bars = ax.barh(y_pos, importances[sorted_idx], color='#34D399', edgecolor='#10B981', height=0.5)
ax.set_yticks(y_pos)
ax.set_yticklabels([X.columns[i] for i in sorted_idx], color='#CBD5E1', fontsize=9)
ax.set_xlabel('Gini Feature Importance', color='#94A3B8', fontsize=9)
ax.set_title('Bab 09: Bobot Fitur Penentu Churn Pelanggan (Random Forest)', color='#F1F5F9', fontsize=10.5, pad=12, fontweight=600)
ax.tick_params(colors='#94A3B8', labelsize=8.5)
for bar in bars:
    val = bar.get_width()
    ax.text(val + 0.01, bar.get_y() + bar.get_height()/2, f'{val*100:.1f}%', va='center', color='#94A3B8', fontsize=8.5)
ax.set_xlim(0, max(importances) * 1.25)
for spine in ax.spines.values():
    spine.set_color('#1E293B')
plt.tight_layout()
plt.savefig('visual_output/09_feature_importance_rf.png', dpi=300)
plt.close()
print('[OK] Visual output Bab 09 berhasil disimpan di visual_output/')
