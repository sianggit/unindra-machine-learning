"""
Module: svm.py
Author: Lab Machine Learning - Universitas Indraprasta PGRI (Unindra)
Description:
    Implementasi kalkulasi Support Vector Machine (Bab 8):
    - Perhitungan lebar margin geometris M = 2 / ||w||
    - Evaluasi jarak sampel ke hyperplane w^T x + b = 0
    - Identifikasi Support Vectors dan evaluasi kernel (Linear vs RBF).
"""

from typing import Dict, Any, Tuple, Optional
import numpy as np


def calculate_svm_margin(weights: np.ndarray) -> Dict[str, Any]:
    """
    Menghitung lebar margin geometris SVM (Bab 8 Halaman 49):
    Margin = 2 / ||w||
    
    Analogi Perkuliahan:
    Seperti membuat jalan tol selebar mungkin antara dua pemukiman warga
    (kelas positif dan kelas negatif) agar tidak ada kendaraan yang bersenggolan.
    """
    w = np.asarray(weights, dtype=float)
    norm_w = float(np.linalg.norm(w))
    if norm_w == 0:
        raise ValueError("Norm vektor bobot w tidak boleh nol.")
    
    margin = float(2.0 / norm_w)
    return {
        "weights": w.tolist(),
        "norm_w (L2)": round(norm_w, 4),
        "margin_width": round(margin, 4),
        "formula": "Margin = 2 / ||w||"
    }


def evaluate_hyperplane_distance(
    x: np.ndarray,
    weights: np.ndarray,
    bias: float
) -> Dict[str, Any]:
    """
    Menghitung jarak tegak lurus suatu titik data ke Hyperplane pemisah:
    distance = |w^T x + b| / ||w||
    """
    x_arr = np.asarray(x, dtype=float)
    w_arr = np.asarray(weights, dtype=float)
    norm_w = np.linalg.norm(w_arr)
    functional_margin = float(np.dot(w_arr, x_arr) + bias)
    geometric_distance = float(abs(functional_margin) / norm_w)

    return {
        "point": x_arr.tolist(),
        "functional_value (w^T x + b)": round(functional_margin, 4),
        "geometric_distance": round(geometric_distance, 4),
        "side": "Positif (+1)" if functional_margin > 0 else ("Negatif (-1)" if functional_margin < 0 else "Pada Hyperplane")
    }
