from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import pandas as pd
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

csv_path = 'ecommerce_sentiment_reviews.csv'
if not os.path.exists(csv_path):
    csv_path = r'd:\Anggit Arisa Siwi Pratama\Unindra\Machine Learning\ml-workspace\2-ml-playground\public\datasets\alternatif\ecommerce_sentiment_reviews.csv'

if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)
    text_col = 'teks_ulasan' if 'teks_ulasan' in df.columns else df.columns[0]
    target_col = 'sentimen_positif' if 'sentimen_positif' in df.columns else df.columns[1]
    texts = df[text_col].astype(str)
    labels = df[target_col].astype(int)
    print(f"[OK] Dataset E-Commerce Sentiment dimuat dari CSV: {df.shape[0]} ulasan.")
else:
    data_reviews = [
        ("barangnya sangat bagus pengiriman cepat sekali respon penjual ramah", 1),
        ("kualitas mantap original sesuai deskripsi toko amanah bintang lima", 1),
        ("packing rapi bubble wrap tebal barang berfungsi normal makasih seller", 1),
        ("sangat memuaskan belanja disini rekomended banget gaes", 1),
        ("produk original berfungsi sangat baik pengiriman kilat", 1),
        ("kecewa berat barang rusak pecah packing asal asalan penjual slow respon", 0),
        ("tidak sesuai deskripsi barang cacat warna salah kurir judes payah", 0),
        ("nyesel beli disini pesanan salah kirim minta retur berbelit belit", 0),
        ("kualitas buruk plastik tipis gampang patah jangan beli nyesel", 0),
        ("lama sekali sampai barang tidak berfungsi rugi uang", 0)
    ] * 20
    df = pd.DataFrame(data_reviews, columns=['teks_ulasan', 'sentimen_positif'])
    texts = df['teks_ulasan']
    labels = df['sentimen_positif']
    print(f"[OK] Dataset E-Commerce Sentiment disintesis: {df.shape[0]} ulasan.")

vec = CountVectorizer(max_features=100)
X = vec.fit_transform(texts)
y = labels

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=42)

bayes = MultinomialNB(alpha=1.0).fit(X_tr, y_tr)
knn = KNeighborsClassifier(n_neighbors=3, weights='distance').fit(X_tr, y_tr)

preds_bayes = bayes.predict(X_te)
preds_knn = knn.predict(X_te)

nb_acc, nb_prec, nb_rec, nb_f1 = accuracy_score(y_te, preds_bayes), precision_score(y_te, preds_bayes), recall_score(y_te, preds_bayes), f1_score(y_te, preds_bayes)
knn_acc, knn_prec, knn_rec, knn_f1 = accuracy_score(y_te, preds_knn), precision_score(y_te, preds_knn), recall_score(y_te, preds_knn), f1_score(y_te, preds_knn)

print(f"\n[Hasil Evaluasi Komparatif Sentimen]")
print(f"1. Multinomial Naive Bayes : {nb_acc*100:.2f}% Akurasi | F1: {nb_f1:.4f}")
print(f"2. Distance-Weighted K-NN  : {knn_acc*100:.2f}% Akurasi | F1: {knn_f1:.4f}")

# Visualisasi Komparasi Bar Chart
plt.style.use('dark_background')
fig, ax = plt.subplots(figsize=(6.5, 3.6), facecolor='#0B0E14')
ax.set_facecolor('#0B0E14')
metrics = ['Akurasi', 'Presisi', 'Recall', 'F1-Score']
knn_scores = [knn_acc*100, knn_prec*100, knn_rec*100, knn_f1*100]
nb_scores = [nb_acc*100, nb_prec*100, nb_rec*100, nb_f1*100]
x = np.arange(len(metrics))
width = 0.35
ax.bar(x - width/2, knn_scores, width, label='K-NN (k=3)', color='#64748B')
ax.bar(x + width/2, nb_scores, width, label='Naive Bayes', color='#34D399')
ax.set_ylabel('Skor (%)', color='#94A3B8', fontsize=9)
ax.set_title('Analisis Sentimen: Benchmark K-NN vs Naive Bayes', color='#F1F5F9', fontsize=10.5, pad=12, fontweight=600)
ax.set_xticks(x)
ax.set_xticklabels(metrics, color='#CBD5E1', fontsize=9)
ax.tick_params(colors='#94A3B8', labelsize=8.5)
ax.set_ylim(50, 105)
ax.legend(loc='lower right', facecolor='#0B0E14', edgecolor='#1E293B', fontsize=8.5)
for spine in ax.spines.values():
    spine.set_color('#1E293B')
plt.tight_layout()
os.makedirs('visual_output', exist_ok=True)
plt.savefig('visual_output/model_benchmark_chart.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print("[OK] Grafik visual disimpan ke visual_output/model_benchmark_chart.png")
