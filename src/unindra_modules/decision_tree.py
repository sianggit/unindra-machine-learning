"""
Module: decision_tree.py
Author: Lab Machine Learning - Universitas Indraprasta PGRI (Unindra)
Description:
    Implementasi penghitung manual Entropy, Gini Impurity, Information Gain (ID3),
    dan Gain Ratio (C4.5) yang krusial untuk UPM/UAS Unindra,
    serta wrapper pelatihan Decision Tree Scikit-Learn dengan Pruning (ccp_alpha).
"""

from typing import List, Dict, Any, Union, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text


def calculate_entropy(y: Union[List[Any], np.ndarray, pd.Series]) -> float:
    """
    Menghitung Entropy Shannon:
    H(S) = - sum(p_i * log2(p_i))
    
    Analogi Perkuliahan:
    Seperti mengukur tingkat kekacauan atau ketidakteraturan isi lemari pakaian.
    Jika semua baju warna putih (homogen), Entropy = 0 (sangat rapi/murni).
    Jika baju terbagi rata 50% putih dan 50% hitam, Entropy = 1 (paling kacau).
    """
    arr = np.asarray(y)
    if len(arr) == 0:
        return 0.0

    _, counts = np.unique(arr, return_counts=True)
    probabilities = counts / len(arr)
    
    # 0 * log2(0) didefinisikan sebagai 0
    entropy = -np.sum([p * np.log2(p) for p in probabilities if p > 0])
    return float(np.round(entropy, 4))


def calculate_gini(y: Union[List[Any], np.ndarray, pd.Series]) -> float:
    """
    Menghitung Gini Impurity (CART):
    Gini(S) = 1 - sum(p_i^2)
    """
    arr = np.asarray(y)
    if len(arr) == 0:
        return 0.0

    _, counts = np.unique(arr, return_counts=True)
    probabilities = counts / len(arr)
    gini = 1.0 - np.sum(probabilities ** 2)
    return float(np.round(gini, 4))


def calculate_information_gain(
    y_parent: Union[List[Any], np.ndarray, pd.Series],
    y_splits: List[Union[List[Any], np.ndarray, pd.Series]],
) -> Dict[str, Any]:
    """
    Menghitung Information Gain:
    Gain(S, A) = Entropy(S) - sum( (|S_v| / |S|) * Entropy(S_v) )
    
    Analogi Perkuliahan:
    Berapa banyak 'kebingungan' yang berhasil dikurangi setelah mahasiswa
    memisahkan dokumen berdasarkan suatu kategori atribut.
    """
    n_parent = len(y_parent)
    if n_parent == 0:
        return {"gain": 0.0, "parent_entropy": 0.0, "sub_entropies": []}

    parent_entropy = calculate_entropy(y_parent)
    weighted_child_entropy = 0.0
    sub_entropies = []

    for child in y_splits:
        n_child = len(child)
        if n_child > 0:
            child_ent = calculate_entropy(child)
            weight = n_child / n_parent
            weighted_child_entropy += weight * child_ent
            sub_entropies.append({
                "count": n_child,
                "weight": round(weight, 4),
                "entropy": child_ent
            })

    gain = parent_entropy - weighted_child_entropy
    return {
        "parent_entropy": round(parent_entropy, 4),
        "weighted_child_entropy": round(weighted_child_entropy, 4),
        "gain": round(gain, 4),
        "sub_details": sub_entropies
    }


def calculate_gain_ratio(
    y_parent: Union[List[Any], np.ndarray, pd.Series],
    y_splits: List[Union[List[Any], np.ndarray, pd.Series]],
) -> Dict[str, Any]:
    """
    Menghitung Gain Ratio C4.5 untuk mengatasi bias Information Gain
    terhadap atribut yang memiliki nilai unik sangat banyak.
    GainRatio(S, A) = Gain(S, A) / SplitInfo(S, A)
    """
    n_parent = len(y_parent)
    gain_info = calculate_information_gain(y_parent, y_splits)
    
    split_info = 0.0
    for child in y_splits:
        p = len(child) / n_parent
        if p > 0:
            split_info -= p * np.log2(p)

    split_info = round(split_info, 4)
    gain = gain_info["gain"]
    ratio = (gain / split_info) if split_info > 0 else 0.0

    return {
        "gain": gain,
        "split_info": split_info,
        "gain_ratio": round(ratio, 4)
    }


def build_decision_tree_classifier(
    criterion: str = "entropy",
    max_depth: Optional[int] = 4,
    min_samples_leaf: int = 5,
    ccp_alpha: float = 0.0,
    random_state: int = 42,
) -> DecisionTreeClassifier:
    """
    Membangun model DecisionTreeClassifier terkalibrasi dengan
    dukungan pre-pruning dan cost-complexity post-pruning.
    """
    return DecisionTreeClassifier(
        criterion=criterion,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        ccp_alpha=ccp_alpha,
        random_state=random_state
    )
