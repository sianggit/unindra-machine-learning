"""
Module: dim_reduction.py
Author: Lab Machine Learning - Universitas Indraprasta PGRI (Unindra)
Description:
    Implementasi Reduksi Dimensi & PCA Manual (Bab 11 Halaman 66):
    - Penghitungan Eigenvalues & Eigenvectors dari matriks kovarians 2x2
    - Penghitungan Explained Variance Ratio untuk tiap komponen utama (PC)
    - Verifikasi hitungan manual ujian UPM Unindra.
"""

from typing import Dict, Any, Tuple
import numpy as np


def solve_pca_2x2_manual(
    cov_matrix: np.ndarray
) -> Dict[str, Any]:
    """
    Menyelesaikan PCA 2x2 langkah demi langkah sesuai Bab 11 Halaman 66 modul Unindra:
    Contoh:
    Sigma = [[2.0, 0.8],
             [0.8, 1.0]]
    det(Sigma - lambda*I) = 0
    (2 - lambda)(1 - lambda) - (0.8 * 0.8) = 0
    lambda^2 - 3*lambda + 1.36 = 0
    lambda_1 = 2.443 (81.4%), lambda_2 = 0.557 (18.6%)
    """
    sigma = np.asarray(cov_matrix, dtype=float)
    if sigma.shape != (2, 2):
        raise ValueError("Matriks kovarians harus berukuran 2x2.")

    a, b = sigma[0, 0], sigma[0, 1]
    c, d = sigma[1, 0], sigma[1, 1]

    # Karakteristik polinomial: lambda^2 - (a+d)*lambda + (ad - bc) = 0
    trace = a + d
    det = a * d - b * c

    # Rumus kuadrat ABC
    discriminant = trace**2 - 4 * det
    lambda_1 = (trace + np.sqrt(discriminant)) / 2.0
    lambda_2 = (trace - np.sqrt(discriminant)) / 2.0

    total_var = lambda_1 + lambda_2
    var_ratio_1 = lambda_1 / total_var
    var_ratio_2 = lambda_2 / total_var

    return {
        "covariance_matrix": sigma.tolist(),
        "trace (total variance)": round(trace, 4),
        "determinant": round(det, 4),
        "characteristic_eq": f"lambda^2 - {trace}*lambda + {det} = 0",
        "lambda_1 (PC1)": round(lambda_1, 4),
        "lambda_2 (PC2)": round(lambda_2, 4),
        "explained_variance_ratio_PC1": round(var_ratio_1, 4),
        "explained_variance_ratio_PC2": round(var_ratio_2, 4),
        "percentage_PC1": f"{var_ratio_1 * 100:.2f}%",
        "percentage_PC2": f"{var_ratio_2 * 100:.2f}%",
    }
