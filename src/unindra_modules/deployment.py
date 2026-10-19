"""
Module: deployment.py
Author: Lab Machine Learning - Universitas Indraprasta PGRI (Unindra)
Description:
    Helper serialisasi model (Joblib) dan template aplikasi Streamlit (Bab 14 Halaman 83).
"""

from typing import Any
from pathlib import Path
import joblib


def export_model_artifact(model: Any, filepath: str) -> str:
    """
    Menyimpan model terlatih dan preprocessing pipeline menggunakan joblib.
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, str(path))
    return str(path)


def load_model_artifact(filepath: str) -> Any:
    """
    Memuat model tersimpan untuk inferensi.
    """
    return joblib.load(str(filepath))
