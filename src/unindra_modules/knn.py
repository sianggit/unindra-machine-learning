"""
Module: knn.py
Author: Lab Machine Learning - Universitas Indraprasta PGRI (Unindra)
Description:
    Implementasi K-Nearest Neighbors (KNN) manual dan scikit-learn,
    termasuk kalkulasi jarak Euclidean & Manhattan bertahap untuk pembuktian
    hitungan manual UPM/UAS, kurva akurasi k, dan visualisasi decision boundary.
"""

from typing import Tuple, List, Dict, Any, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class ManualKNNClassifier:
    """
    Implementasi K-Nearest Neighbors dari Scratch (Numpy).
    
    Cocok untuk membongkar 'Black Box' algoritma dan membuktikan hasil
    perhitungan manual pada lembar ujian UPM/UAS Unindra.
    """
    def __init__(self, k: int = 3, metric: str = "euclidean"):
        self.k = k
        self.metric = metric.lower()
        self.X_train: Optional[np.ndarray] = None
        self.y_train: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ManualKNNClassifier":
        self.X_train = np.asarray(X, dtype=float)
        self.y_train = np.asarray(y)
        return self

    def _compute_distance(self, x1: np.ndarray, x2: np.ndarray) -> float:
        if self.metric == "euclidean":
            return float(np.sqrt(np.sum((x1 - x2) ** 2)))
        elif self.metric == "manhattan":
            return float(np.sum(np.abs(x1 - x2)))
        else:
            raise ValueError(f"Metrik jarak '{self.metric}' tidak dikenal.")

    def explain_prediction(self, x_query: np.ndarray) -> Dict[str, Any]:
        """
        Menghasilkan rincian perhitungan manual langkah demi langkah:
        1. Jarak ke setiap data training.
        2. K tetangga terdekat terurut.
        3. Frekuensi voting kelas.
        """
        x_q = np.asarray(x_query, dtype=float)
        distances = []
        for idx, x_tr in enumerate(self.X_train):
            d = self._compute_distance(x_q, x_tr)
            distances.append({
                "train_index": idx,
                "train_sample": x_tr.tolist(),
                "target_class": self.y_train[idx],
                "distance": round(d, 4),
            })

        sorted_neighbors = sorted(distances, key=lambda item: item["distance"])
        top_k = sorted_neighbors[:self.k]

        votes: Dict[Any, int] = {}
        for item in top_k:
            cls = item["target_class"]
            votes[cls] = votes.get(cls, 0) + 1

        predicted_class = max(votes, key=votes.get)

        return {
            "query_sample": x_q.tolist(),
            "metric": self.metric,
            "k": self.k,
            "top_k_neighbors": top_k,
            "vote_distribution": votes,
            "predicted_class": predicted_class,
        }

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_arr = np.asarray(X, dtype=float)
        predictions = [self.explain_prediction(x)["predicted_class"] for x in X_arr]
        return np.array(predictions)


def evaluate_knn_performance(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    k_range: range = range(1, 26),
) -> pd.DataFrame:
    """
    Mengevaluasi akurasi Train vs Test terhadap variasi nilai k.
    Membuktikan fenomena Overfitting (k=1) dan Underfitting (k besar).
    """
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import accuracy_score

    results = []
    for k in k_range:
        clf = KNeighborsClassifier(n_neighbors=k)
        clf.fit(X_train, y_train)
        acc_tr = accuracy_score(y_train, clf.predict(X_train))
        acc_te = accuracy_score(y_test, clf.predict(X_test))
        results.append({"k": k, "train_accuracy": acc_tr, "test_accuracy": acc_te})

    return pd.DataFrame(results)


def plot_decision_boundaries(
    clf: Any,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: Tuple[str, str],
    title: str = "KNN Decision Boundary",
    save_path: Optional[str] = None,
) -> None:
    """
    Membuat plot bidang keputusan (Decision Boundary) 2D.
    """
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.05), np.arange(y_min, y_max, 0.05))

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap="coolwarm")
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", edgecolors="k", s=40)
    plt.xlabel(feature_names[0])
    plt.ylabel(feature_names[1])
    plt.title(title, fontsize=12, fontweight="bold")
    plt.grid(True, linestyle="--", alpha=0.5)

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close()
