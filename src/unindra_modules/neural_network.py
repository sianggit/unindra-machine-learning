"""
Module: neural_network.py
Author: Lab Machine Learning - Universitas Indraprasta PGRI (Unindra)
Description:
    Implementasi Dasar Neural Network / MLP (Bab 12 Halaman 72):
    - Perhitungan manual forward pass satu neuron dengan fungsi aktivasi (ReLU, Sigmoid)
    - Verifikasi hitungan lembar modul praktikum Unindra:
      Input x = [0.5, -0.2, 0.8], Bobot w = [0.4, 0.6, -0.3], Bias b = 0.1
      z = -0.06 -> a = ReLU(z) = 0
"""

from typing import Dict, Any, List
import numpy as np


def neuron_forward_pass_manual(
    inputs: List[float],
    weights: List[float],
    bias: float,
    activation: str = "relu"
) -> Dict[str, Any]:
    """
    Menghitung forward pass manual satu neuron sesuai Bab 12 Halaman 72.
    """
    x = np.asarray(inputs, dtype=float)
    w = np.asarray(weights, dtype=float)
    b = float(bias)

    dot_products = (x * w).tolist()
    z = float(np.sum(x * w) + b)

    if activation.lower() == "relu":
        a = float(max(0.0, z))
    elif activation.lower() == "sigmoid":
        a = float(1.0 / (1.0 + np.exp(-z)))
    else:
        raise ValueError(f"Fungsi aktivasi '{activation}' tidak didukung.")

    return {
        "inputs": x.tolist(),
        "weights": w.tolist(),
        "bias": b,
        "weighted_products": [round(val, 4) for val in dot_products],
        "linear_sum_z": round(z, 4),
        "activation_function": activation,
        "output_a": round(a, 4),
        "neuron_status": "Aktif" if a > 0 else "Inaktif / Mati (ReLU Output = 0)"
    }
