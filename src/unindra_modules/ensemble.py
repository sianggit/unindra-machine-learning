"""
Module: ensemble.py
Author: Lab Machine Learning - Universitas Indraprasta PGRI (Unindra)
Description:
    Implementasi Ensemble Learning (Bab 9):
    - Pembuktian probabilitas Out-of-Bag (OOB): (1 - 1/n)^n -> 1/e ~ 0.368
    - Feature Importance aggregator & Voting Classifier simulation.
"""

from typing import Dict, Any, List
import numpy as np


def calculate_oob_probability(n_samples: int) -> Dict[str, Any]:
    """
    Menghitung probabilitas suatu sampel tidak terpilih dalam bootstrap sampling (OOB):
    P(not selected) = (1 - 1/n)^n
    Untuk n besar, mendekati 1/e approx 0.367879 (36.8%).
    """
    if n_samples <= 0:
        raise ValueError("n_samples harus lebih besar dari 0.")
    
    prob_single_draw_not_picked = (n_samples - 1) / n_samples
    prob_oob = float(prob_single_draw_not_picked ** n_samples)
    theoretical_limit = float(1.0 / np.e)

    return {
        "n_samples": n_samples,
        "prob_oob": round(prob_oob, 6),
        "percentage_oob": round(prob_oob * 100, 2),
        "theoretical_limit (1/e)": round(theoretical_limit, 6),
        "difference": round(abs(prob_oob - theoretical_limit), 6)
    }
