"""
Module: kmeans.py
Author: Lab Machine Learning - Universitas Indraprasta PGRI (Unindra)
Description:
    Implementasi K-Means Clustering langkah demi langkah (1D & 2D) untuk pembuktian
    kalkulasi manual ujian UPM/UAS Unindra (perhitungan jarak & update centroid),
    serta analisis WCSS Elbow Method dan evaluasi Silhouette Score.
"""

from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class ManualKMeans1D:
    """
    Simulasi K-Means 1 Dimensi langkah demi langkah.
    Sesuai persis dengan tipe soal hitungan lembar UPM/UAS Unindra
    (contoh: data [2, 3, 8, 9, 10] dengan centroid awal [2, 9]).
    """
    def __init__(self, initial_centroids: List[float], max_iter: int = 10):
        self.initial_centroids = [float(c) for c in initial_centroids]
        self.max_iter = max_iter
        self.history: List[Dict[str, Any]] = []

    def fit(self, data: List[float]) -> "ManualKMeans1D":
        data_arr = np.asarray(data, dtype=float)
        centroids = np.asarray(self.initial_centroids, dtype=float)
        self.history = []

        for it in range(1, self.max_iter + 1):
            step_record = {
                "iteration": it,
                "current_centroids": centroids.tolist(),
                "point_details": [],
                "clusters": {i: [] for i in range(len(centroids))},
            }

            # Hitung jarak dan tentukan klaster
            for x in data_arr:
                distances = [abs(x - c) for c in centroids]
                assigned_cluster = int(np.argmin(distances))
                step_record["clusters"][assigned_cluster].append(float(x))
                step_record["point_details"].append({
                    "x": float(x),
                    "distances_to_centroids": [round(d, 4) for d in distances],
                    "assigned_cluster": assigned_cluster + 1,  # 1-indexed untuk mahasiswa
                })

            # Update Centroid baru
            new_centroids = []
            for i in range(len(centroids)):
                pts = step_record["clusters"][i]
                if len(pts) > 0:
                    new_c = float(np.mean(pts))
                else:
                    new_c = centroids[i]
                new_centroids.append(round(new_c, 4))

            step_record["new_centroids"] = new_centroids
            self.history.append(step_record)

            # Cek Konvergensi
            if np.allclose(centroids, new_centroids, atol=1e-5):
                step_record["converged"] = True
                break
            else:
                step_record["converged"] = False
                centroids = np.array(new_centroids)

        return self


class ManualKMeans2D:
    """
    Simulasi K-Means 2D dengan jarak Euclidean untuk visualisasi koordinat.
    """
    def __init__(self, n_clusters: int = 3, max_iter: int = 100, random_state: int = 42):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state
        self.centroids: Optional[np.ndarray] = None
        self.labels_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> "ManualKMeans2D":
        np.random.seed(self.random_state)
        X_arr = np.asarray(X, dtype=float)
        n_samples = len(X_arr)

        # Inisialisasi acak
        random_idx = np.random.choice(n_samples, self.n_clusters, replace=False)
        self.centroids = X_arr[random_idx].copy()

        for _ in range(self.max_iter):
            # Jarak Euclidean ke semua centroid
            distances = np.linalg.norm(X_arr[:, np.newaxis] - self.centroids, axis=2)
            new_labels = np.argmin(distances, axis=1)

            # Update Centroid
            new_centroids = np.array([
                X_arr[new_labels == k].mean(axis=0) if np.sum(new_labels == k) > 0 else self.centroids[k]
                for k in range(self.n_clusters)
            ])

            if np.allclose(self.centroids, new_centroids, atol=1e-4):
                break
            self.centroids = new_centroids

        self.labels_ = new_labels
        return self


def calculate_wcss(X: np.ndarray, centroids: np.ndarray, labels: np.ndarray) -> float:
    """
    Menghitung Within-Cluster Sum of Squares (Inertia/WCSS):
    WCSS = sum_k sum_{x in C_k} ||x - mu_k||^2
    """
    wcss = 0.0
    for k in range(len(centroids)):
        cluster_pts = X[labels == k]
        if len(cluster_pts) > 0:
            wcss += np.sum((cluster_pts - centroids[k]) ** 2)
    return float(np.round(wcss, 4))


def elbow_method_analysis(
    X: np.ndarray,
    k_range: range = range(1, 10),
    save_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Menjalankan K-Means untuk rentang k dan memvisualisasikan grafik siku (Elbow Curve).
    """
    from sklearn.cluster import KMeans

    results = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X)
        results.append({"k": k, "wcss": km.inertia_})

    df_results = pd.DataFrame(results)

    plt.figure(figsize=(8, 5))
    plt.plot(df_results["k"], df_results["wcss"], marker="o", color="#1e3a8a", linewidth=2)
    plt.title("Elbow Method untuk Menentukan Jumlah Klaster Optimal (k)", fontsize=12, fontweight="bold")
    plt.xlabel("Jumlah Klaster (k)")
    plt.ylabel("Within-Cluster Sum of Squares (WCSS)")
    plt.grid(True, linestyle="--", alpha=0.5)

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close()

    return df_results
