import os
import sys
import io
import time
import json
import shutil
from pathlib import Path

# Ensure UTF-8 output on Windows console
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    except Exception:
        pass
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    auc,
    precision_recall_curve,
    confusion_matrix,
    classification_report
)
import joblib

# Konfigurasi path
CURRENT_DIR = Path(__file__).resolve().parent
DATA_DIR = CURRENT_DIR / "data"
MODELS_DIR = CURRENT_DIR / "models"
VISUAL_DIR = CURRENT_DIR / "visual_output"
PUBLIC_VISUAL_DIR = CURRENT_DIR.parent.parent / "2-ml-playground" / "public" / "visual_output" / "bab-14"

DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
VISUAL_DIR.mkdir(parents=True, exist_ok=True)
PUBLIC_VISUAL_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 65)
print("[BAB 14] Mini Project End-to-End: Komparasi Model dan Deployment")
print("   Laboratorium Machine Learning - Universitas Indraprasta PGRI")
print("   Mahasiswa: Anggit Arisa Siwi Pratama (NPM: 202343570045)")
print("=" * 65)

# ─── 1. Business Understanding & Akuisisi Dataset ─────────────────────────────
# Kasus: Prediksi Risiko Gagal Bayar Pinjaman (Loan Default Prediction)
np.random.seed(42)
n_samples = 1200

# Sintesis fitur yang realistis dengan dinamika perbankan Indonesia
pendapatan = np.random.lognormal(mean=16.0, sigma=0.5, size=n_samples) # Rp 5jt - 30jt+
pendapatan = np.clip(np.round(pendapatan, -5), 3_500_000, 45_000_000)

usia = np.random.randint(21, 62, size=n_samples)
lama_kerja = np.clip((usia - 21) - np.random.randint(0, 10, size=n_samples), 0, 35)

jumlah_pinjaman = np.random.choice([15_000_000, 30_000_000, 50_000_000, 100_000_000, 150_000_000], size=n_samples, p=[0.3, 0.3, 0.2, 0.15, 0.05])
cicilan_bulanan = jumlah_pinjaman / np.random.choice([12, 24, 36, 48], size=n_samples)
rasio_cicilan_pendapatan = cicilan_bulanan / pendapatan

skor_kredit = np.random.normal(loc=650, scale=80, size=n_samples)
skor_kredit = np.clip(np.round(skor_kredit), 350, 850)

# Formulasi log-odds probabilitas default
logit = (
    - 2.8
    + 3.5 * (rasio_cicilan_pendapatan > 0.40)
    + 1.8 * (rasio_cicilan_pendapatan > 0.30)
    - 0.008 * (skor_kredit - 600)
    - 0.08 * lama_kerja
    + 0.6 * (usia < 25)
    + np.random.normal(0, 0.4, size=n_samples)
)
prob_default = 1 / (1 + np.exp(-logit))
gagal_bayar = (np.random.uniform(0, 1, size=n_samples) < prob_default).astype(int)

df_loan = pd.DataFrame({
    "pendapatan_bulanan": pendapatan,
    "usia": usia,
    "lama_kerja": lama_kerja,
    "jumlah_pinjaman": jumlah_pinjaman,
    "rasio_cicilan": np.round(rasio_cicilan_pendapatan, 3),
    "skor_kredit": skor_kredit,
    "status_default": gagal_bayar
})

csv_path = DATA_DIR / "loan_default_dataset.csv"
df_loan.to_csv(csv_path, index=False)
print(f"\n[Fase 1: Data Understanding]")
print(f"- Dataset Kredit tersimpan di: {csv_path.name}")
print(f"- Dimensi Data: {df_loan.shape[0]} baris, {df_loan.shape[1]} kolom")
print(f"- Distribusi Kelas Target (status_default):")
counts = df_loan["status_default"].value_counts()
print(f"  * Lancar (0)      : {counts[0]} ({counts[0]/len(df_loan)*100:.1f}%)")
print(f"  * Gagal Bayar (1) : {counts[1]} ({counts[1]/len(df_loan)*100:.1f}%)")

# ─── 2. Data Preparation: Split Data Latih & Uji ─────────────────────────────
feature_cols = ["pendapatan_bulanan", "usia", "lama_kerja", "jumlah_pinjaman", "rasio_cicilan", "skor_kredit"]
X = df_loan[feature_cols]
y = df_loan["status_default"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
print(f"\n[Fase 2: Data Preparation]")
print(f"- Data Latih : {X_train.shape[0]} sampel (Stratified 80%)")
print(f"- Data Uji   : {X_test.shape[0]} sampel (Stratified 20%)")

# ─── 3. Modeling: Komparasi 3 Algoritma & Tuning Sistematis ──────────────────
print(f"\n[Fase 3: Modeling & Tuning 3 Algoritma]")

model_definitions = {
    "Random Forest": {
        "model": RandomForestClassifier(random_state=42),
        "params": {
            "n_estimators": [50, 100, 200],
            "max_depth": [4, 6, 10, None],
            "min_samples_leaf": [1, 2, 4],
            "class_weight": ["balanced", None]
        },
        "interpretability": 3 # Skala 1-5
    },
    "Gradient Boosting": {
        "model": GradientBoostingClassifier(random_state=42),
        "params": {
            "n_estimators": [50, 100, 150],
            "learning_rate": [0.03, 0.08, 0.15],
            "max_depth": [3, 4, 5],
            "subsample": [0.8, 1.0]
        },
        "interpretability": 2
    },
    "SVM (RBF Kernel)": {
        "model": Pipeline([
            ("scaler", StandardScaler()),
            ("svm", SVC(probability=True, random_state=42))
        ]),
        "params": {
            "svm__C": [0.5, 1.0, 5.0, 10.0],
            "svm__gamma": ["scale", "auto", 0.05, 0.1],
            "svm__class_weight": ["balanced", None]
        },
        "interpretability": 2
    }
}

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
benchmark_results = {}

for name, config in model_definitions.items():
    print(f"\n[Proses] Melatih dan melakukan tuning: {name}...")
    t0 = time.time()
    search = RandomizedSearchCV(
        config["model"],
        param_distributions=config["params"],
        n_iter=6,
        cv=skf,
        scoring="f1",
        random_state=42,
        n_jobs=-1
    )
    search.fit(X_train, y_train)
    t_train = time.time() - t0
    
    best_est = search.best_estimator_
    
    # Latency inferensi per 100 sampel
    t_inf_start = time.time()
    for _ in range(5):
        best_est.predict(X_test)
    t_latency_ms = ((time.time() - t_inf_start) / (5 * len(X_test))) * 1000
    
    y_pred = best_est.predict(X_test)
    y_prob = best_est.predict_proba(X_test)[:, 1]
    
    f1_cv = search.best_score_
    test_acc = accuracy_score(y_test, y_pred)
    test_prec = precision_score(y_test, y_pred, zero_division=0)
    test_rec = recall_score(y_test, y_pred, zero_division=0)
    test_f1 = f1_score(y_test, y_pred, zero_division=0)
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    test_roc_auc = auc(fpr, tpr)
    prec_arr, rec_arr, _ = precision_recall_curve(y_test, y_prob)
    test_pr_auc = auc(rec_arr, prec_arr)
    
    # Skor Gabungan Multi-Kriteria (Sub-bab 14.9)
    # Skor = 0.7 * F1 + 0.3 * (Interpretability / 5)
    combined_score = 0.7 * test_f1 + 0.3 * (config["interpretability"] / 5.0)
    
    benchmark_results[name] = {
        "best_estimator": best_est,
        "best_params": search.best_params_,
        "cv_f1": f1_cv,
        "test_acc": test_acc,
        "test_prec": test_prec,
        "test_rec": test_rec,
        "test_f1": test_f1,
        "test_roc_auc": test_roc_auc,
        "test_pr_auc": test_pr_auc,
        "train_time_sec": t_train,
        "latency_ms": t_latency_ms,
        "interpretability": config["interpretability"],
        "combined_score": combined_score,
        "y_pred": y_pred,
        "y_prob": y_prob
    }
    
    print(f"   - CV F1-Score Terbaik : {f1_cv:.4f}")
    print(f"   - Test F1-Score       : {test_f1:.4f} | Recall: {test_rec:.4f} | PR-AUC: {test_pr_auc:.4f}")
    print(f"   - Skor Gabungan       : {combined_score:.4f} (Interpretability: {config['interpretability']}/5)")
    print(f"   - Waktu Latih: {t_train:.2f}s | Latensi Inferensi: {t_latency_ms:.3f} ms/sampel")

# ─── 4. Evaluation & Pemilihan Model Terbaik ─────────────────────────────────
print(f"\n[Fase 4: Rekapitulasi Evaluasi & Pemilihan Model]")
df_benchmark = pd.DataFrame([
    {
        "Model": name,
        "CV F1": f"{res['cv_f1']*100:.2f}%",
        "Test F1": f"{res['test_f1']*100:.2f}%",
        "Recall": f"{res['test_rec']*100:.2f}%",
        "ROC-AUC": f"{res['test_roc_auc']:.4f}",
        "Latensi (ms)": f"{res['latency_ms']:.3f}",
        "Interp (1-5)": res["interpretability"],
        "Skor Gabungan": f"{res['combined_score']:.4f}"
    }
    for name, res in benchmark_results.items()
])
print(df_benchmark.to_string(index=False))

# Pilih model dengan skor gabungan tertinggi
best_model_name = max(benchmark_results, key=lambda k: benchmark_results[k]["combined_score"])
best_model_data = benchmark_results[best_model_name]
best_estimator = best_model_data["best_estimator"]

print(f"\n[Hasil] Model terpilih berdasarkan kriteria multi-evaluasi: {best_model_name}")
print(f"   Parameter Terbaik: {best_model_data['best_params']}")
print(f"   Skor Gabungan    : {best_model_data['combined_score']:.4f}")

# ─── 5. Deployment: Serialisasi Model & Ekspor Bobot Ringan ──────────────────
print(f"\n[Fase 5: Serialisasi Artefak Model & Integrasi Vercel]")
model_artifact_path = MODELS_DIR / "model_loan_default_best.joblib"
joblib.dump(best_estimator, model_artifact_path)
print(f"- Artefak Joblib tersimpan di: {model_artifact_path}")

# Verifikasi pemuatan artefak
loaded_model = joblib.load(model_artifact_path)
test_sample = X_test.iloc[:1]
test_pred = loaded_model.predict(test_sample)[0]
test_prob = loaded_model.predict_proba(test_sample)[0][1]
print(f"- Uji Coba Pemuatan Artefak Sukses: Prediksi Kelas {test_pred} (Probabilitas Default: {test_prob*100:.1f}%)")

# Ekspor parameter bobot & normalisasi ke JSON agar dapat dijalankan secara instan di React Simulator (Vercel)
# Hitung statistik normalisasi fitur untuk client-side normalizer
scaler_stats = {
    "means": {col: float(X_train[col].mean()) for col in feature_cols},
    "stds": {col: float(X_train[col].std()) for col in feature_cols}
}

# Ambil feature importance jika model berbasis ensemble
feat_importances = {}
if hasattr(best_estimator, "feature_importances_"):
    feat_importances = {col: float(val) for col, val in zip(feature_cols, best_estimator.feature_importances_)}
elif best_model_name == "Random Forest" and hasattr(best_estimator, "estimators_"):
    feat_importances = {col: float(val) for col, val in zip(feature_cols, best_estimator.feature_importances_)}
else:
    # Default dari Random Forest benchmark
    rf_est = benchmark_results["Random Forest"]["best_estimator"]
    feat_importances = {col: float(val) for col, val in zip(feature_cols, rf_est.feature_importances_)}

client_weights_path = DATA_DIR / "loan_default_weights.json"
client_weights_data = {
    "best_model_name": best_model_name,
    "combined_score": round(float(best_model_data["combined_score"]), 4),
    "feature_names": feature_cols,
    "scaler_stats": scaler_stats,
    "feature_importances": feat_importances,
    "benchmark_summary": [
        {
            "name": name,
            "cv_f1": round(float(res["cv_f1"]), 4),
            "test_f1": round(float(res["test_f1"]), 4),
            "recall": round(float(res["test_rec"]), 4),
            "roc_auc": round(float(res["test_roc_auc"]), 4),
            "interpretability": res["interpretability"],
            "combined_score": round(float(res["combined_score"]), 4)
        }
        for name, res in benchmark_results.items()
    ]
}

with open(client_weights_path, "w", encoding="utf-8") as f:
    json.dump(client_weights_data, f, indent=2)
print(f"- Metadata inferensi Vercel tersimpan di: {client_weights_path.name}")

# ─── 6. Ekspor Grafik Publikasi 300 DPI ───────────────────────────────────────
print(f"\n[Fase 6: Pembuatan Grafik Analitis Resolusi Tinggi (300 DPI)]")
plt.style.use("dark_background")
DARK_BG = "#0B0E14"
PANEL_BG = "#131823"
ACCENT_BLUE = "#38BDF8"
ACCENT_GREEN = "#10B981"
ACCENT_PURPLE = "#818CF8"
ACCENT_AMBER = "#F59E0B"
ACCENT_RED = "#EF4444"

# 6.1. Bar Chart Komparasi F1-Score & Skor Gabungan
fig, ax = plt.subplots(figsize=(7, 4.5), facecolor=DARK_BG)
ax.set_facecolor(PANEL_BG)

models = list(benchmark_results.keys())
x_pos = np.arange(len(models))
width = 0.35

cv_f1s = [benchmark_results[m]["cv_f1"] * 100 for m in models]
scores = [benchmark_results[m]["combined_score"] * 100 for m in models]

rects1 = ax.bar(x_pos - width/2, cv_f1s, width, label="CV F1-Score", color=ACCENT_BLUE, alpha=0.9, edgecolor="#0284C7")
rects2 = ax.bar(x_pos + width/2, scores, width, label="Skor Gabungan (0.7*F1 + 0.3*Interp)", color=ACCENT_GREEN, alpha=0.9, edgecolor="#059669")

ax.set_ylabel("Persentase Skor (%)", color="#94A3B8", fontsize=10)
ax.set_title("Bab 14: Perbandingan Performa & Skor Multi-Kriteria 3 Model", color="#F1F5F9", fontsize=11, fontweight="bold", pad=12)
ax.set_xticks(x_pos)
ax.set_xticklabels(models, color="#E2E8F0", fontsize=9.5)
ax.set_ylim(0, 100)
ax.legend(facecolor=PANEL_BG, edgecolor="#334155", labelcolor="#F1F5F9", fontsize=8.5)
ax.grid(color="#334155", linestyle=":", alpha=0.5, axis="y")

for rect in rects1:
    h = rect.get_height()
    ax.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", color="#38BDF8", fontsize=8, fontweight="bold")
for rect in rects2:
    h = rect.get_height()
    ax.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", color="#10B981", fontsize=8, fontweight="bold")

plt.tight_layout()
p1 = VISUAL_DIR / "14_model_f1_benchmark_comparison.png"
plt.savefig(p1, dpi=300)
shutil.copy(p1, PUBLIC_VISUAL_DIR / p1.name)
plt.close()
print(f"- Grafik 1 tersimpan: {p1.name}")

# 6.2. Radar Chart Trade-off Multi-Kriteria
labels = ["F1-Score", "Recall", "ROC-AUC", "Interpretability", "Kecepatan (1/Latency)"]
num_vars = len(labels)
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1]

fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True), facecolor=DARK_BG)
ax.set_facecolor(PANEL_BG)

colors = [ACCENT_BLUE, ACCENT_GREEN, ACCENT_PURPLE]
for idx, (name, res) in enumerate(benchmark_results.items()):
    # Normalisasi kecepatan: semakin kecil latensi ms, semakin tinggi nilainya
    speed_score = min(1.0, 0.05 / max(0.001, res["latency_ms"]))
    values = [
        res["test_f1"],
        res["test_rec"],
        res["test_roc_auc"],
        res["interpretability"] / 5.0,
        speed_score
    ]
    values += values[:1]
    ax.plot(angles, values, color=colors[idx], linewidth=2, label=name)
    ax.fill(angles, values, color=colors[idx], alpha=0.15)

ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)
ax.set_thetagrids(np.degrees(angles[:-1]), labels, color="#94A3B8", fontsize=8.5)
ax.set_ylim(0, 1.05)
ax.set_title("Bab 14: Radar Trade-Off Multi-Kriteria Pemilihan Model", color="#F1F5F9", fontsize=11, fontweight="bold", pad=18)
ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), facecolor=PANEL_BG, edgecolor="#334155", labelcolor="#F1F5F9", fontsize=8)
ax.grid(color="#334155", linestyle=":", alpha=0.6)

plt.tight_layout()
p2 = VISUAL_DIR / "14_multicriteria_decision_radar.png"
plt.savefig(p2, dpi=300)
shutil.copy(p2, PUBLIC_VISUAL_DIR / p2.name)
plt.close()
print(f"- Grafik 2 tersimpan: {p2.name}")

# 6.3. Confusion Matrix Model Terbaik
cm = confusion_matrix(y_test, best_model_data["y_pred"])
fig, ax = plt.subplots(figsize=(5, 4.2), facecolor=DARK_BG)
ax.set_facecolor(PANEL_BG)
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
    annot_kws={"size": 13, "weight": "bold", "color": "#FFFFFF"},
    xticklabels=["Lancar (0)", "Default (1)"],
    yticklabels=["Lancar (0)", "Default (1)"]
)
ax.set_title(f"Confusion Matrix: {best_model_name} (F1={best_model_data['test_f1']*100:.1f}%)", color="#F1F5F9", fontsize=10.5, pad=12, fontweight="bold")
ax.set_xlabel("Label Prediksi Model", color="#94A3B8", fontsize=9)
ax.set_ylabel("Label Aktual Nasabah", color="#94A3B8", fontsize=9)
ax.tick_params(colors="#94A3B8", labelsize=8.5)

plt.tight_layout()
p3 = VISUAL_DIR / "14_best_model_confusion_matrix.png"
plt.savefig(p3, dpi=300)
shutil.copy(p3, PUBLIC_VISUAL_DIR / p3.name)
plt.close()
print(f"- Grafik 3 tersimpan: {p3.name}")

# 6.4. Dual ROC & PR Curves
fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(10, 4.2), facecolor=DARK_BG)
ax_roc.set_facecolor(PANEL_BG)
ax_pr.set_facecolor(PANEL_BG)

fpr, tpr, _ = roc_curve(y_test, best_model_data["y_prob"])
roc_auc_val = auc(fpr, tpr)
ax_roc.plot(fpr, tpr, color=ACCENT_BLUE, lw=2.2, label=f"{best_model_name} (AUC = {roc_auc_val:.3f})")
ax_roc.plot([0, 1], [0, 1], color="#64748B", lw=1.2, linestyle="--", label="Random Chance (0.50)")
ax_roc.set_title("Kurva ROC (Receiver Operating Characteristic)", color="#F1F5F9", fontsize=9.5, fontweight="bold")
ax_roc.set_xlabel("False Positive Rate", color="#94A3B8", fontsize=8.5)
ax_roc.set_ylabel("True Positive Rate (Recall)", color="#94A3B8", fontsize=8.5)
ax_roc.legend(facecolor=PANEL_BG, edgecolor="#334155", labelcolor="#F1F5F9", fontsize=8)
ax_roc.grid(color="#334155", linestyle=":", alpha=0.5)

prec_arr, rec_arr, _ = precision_recall_curve(y_test, best_model_data["y_prob"])
pr_auc_val = auc(rec_arr, prec_arr)
baseline_pr = y_test.sum() / len(y_test)
ax_pr.plot(rec_arr, prec_arr, color=ACCENT_GREEN, lw=2.2, label=f"PR Curve (AUC = {pr_auc_val:.3f})")
ax_pr.axhline(baseline_pr, color="#F59E0B", lw=1.2, linestyle="--", label=f"Baseline Proporsi ({baseline_pr:.2f})")
ax_pr.set_title("Kurva Precision-Recall (Fokus Kasus Default)", color="#F1F5F9", fontsize=9.5, fontweight="bold")
ax_pr.set_xlabel("Recall (Sensitivitas)", color="#94A3B8", fontsize=8.5)
ax_pr.set_ylabel("Precision", color="#94A3B8", fontsize=8.5)
ax_pr.legend(facecolor=PANEL_BG, edgecolor="#334155", labelcolor="#F1F5F9", fontsize=8)
ax_pr.grid(color="#334155", linestyle=":", alpha=0.5)

plt.tight_layout()
p4 = VISUAL_DIR / "14_best_model_roc_pr_curves.png"
plt.savefig(p4, dpi=300)
shutil.copy(p4, PUBLIC_VISUAL_DIR / p4.name)
plt.close()
print(f"- Grafik 4 tersimpan: {p4.name}")

# 6.5. Feature Importance Horizontal Bar
fig, ax = plt.subplots(figsize=(7, 4), facecolor=DARK_BG)
ax.set_facecolor(PANEL_BG)

sorted_feats = sorted(feat_importances.items(), key=lambda x: x[1])
feat_keys = [x[0].replace("_", " ").title() for x in sorted_feats]
feat_vals = [x[1] for x in sorted_feats]

bars = ax.barh(feat_keys, feat_vals, color=ACCENT_PURPLE, alpha=0.85, edgecolor="#6366F1")
ax.set_xlabel("Mean Decrease in Impurity (Gini Importance)", color="#94A3B8", fontsize=9)
ax.set_title(f"Bab 14: Kepentingan Fitur Penentu Risiko Kredit ({best_model_name})", color="#F1F5F9", fontsize=10.5, fontweight="bold", pad=12)
ax.tick_params(colors="#94A3B8", labelsize=8.5)
ax.grid(color="#334155", linestyle=":", alpha=0.5, axis="x")

for bar in bars:
    w = bar.get_width()
    ax.annotate(f"{w:.3f}", xy=(w, bar.get_y() + bar.get_height() / 2), xytext=(4, 0), textcoords="offset points", ha="left", va="center", color="#E2E8F0", fontsize=8, fontweight="bold")

plt.tight_layout()
p5 = VISUAL_DIR / "14_feature_importance_analysis.png"
plt.savefig(p5, dpi=300)
shutil.copy(p5, PUBLIC_VISUAL_DIR / p5.name)
plt.close()
print(f"- Grafik 5 tersimpan: {p5.name}")

print("\n" + "=" * 65)
print("[Selesai] Pipeline Bab 14 berhasil dieksekusi secara lengkap.")
print(f"   Model Terbaik: {best_model_name} (Tersimpan di: models/{model_artifact_path.name})")
print(f"   Visual Output: 5 Grafik tersimpan di visual_output/ dan disinkronkan ke Vercel public/")
print("=" * 65)
