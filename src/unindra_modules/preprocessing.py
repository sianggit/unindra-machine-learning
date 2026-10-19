"""
Module: preprocessing.py
Author: Lab Machine Learning - Universitas Indraprasta PGRI (Unindra)
Description:
    Pipeline pembersihan data, penanganan missing values (KNNImputer),
    deteksi anomali/outlier (IsolationForest), Feature Scaling & Encoding (ColumnTransformer),
    serta strategi mitigasi data tidak seimbang (Imbalanced Classification).
"""

from typing import List, Optional, Tuple, Dict, Any
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score, accuracy_score


def handle_missing_values(
    df: pd.DataFrame,
    numeric_cols: Optional[List[str]] = None,
    strategy: str = "knn",
    n_neighbors: int = 5,
) -> pd.DataFrame:
    """
    Menangani missing values dengan metode KNNImputer atau SimpleImputer.
    
    Analogi Perkuliahan:
    Seperti menebak preferensi makan siang seorang mahasiswa yang lupa mengisi formulir
    berdasarkan 5 teman terdekatnya yang punya pola belanja serupa.
    """
    df_cleaned = df.copy()
    if numeric_cols is None:
        numeric_cols = df_cleaned.select_dtypes(include=[np.number]).columns.tolist()

    if not numeric_cols:
        return df_cleaned

    if strategy == "knn":
        imputer = KNNImputer(n_neighbors=n_neighbors)
        df_cleaned[numeric_cols] = imputer.fit_transform(df_cleaned[numeric_cols])
    elif strategy == "median":
        imputer = SimpleImputer(strategy="median")
        df_cleaned[numeric_cols] = imputer.fit_transform(df_cleaned[numeric_cols])
    elif strategy == "mean":
        imputer = SimpleImputer(strategy="mean")
        df_cleaned[numeric_cols] = imputer.fit_transform(df_cleaned[numeric_cols])
    else:
        raise ValueError(f"Strategi '{strategy}' tidak didukung. Pilih 'knn', 'median', atau 'mean'.")

    return df_cleaned


def detect_and_handle_outliers(
    df: pd.DataFrame,
    numeric_cols: Optional[List[str]] = None,
    contamination: float = 0.02,
    random_state: int = 42,
    drop: bool = True,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Mendeteksi anomali/pencilan menggunakan Isolation Forest.
    
    Analogi Perkuliahan:
    Seperti satpam kampus yang memisahkan oknum yang berperilaku sangat aneh
    hanya dengan beberapa pertanyaan pohon keputusan sederhana.
    """
    df_result = df.copy()
    if numeric_cols is None:
        numeric_cols = df_result.select_dtypes(include=[np.number]).columns.tolist()

    iso = IsolationForest(
        contamination=contamination,
        random_state=random_state,
        n_estimators=100
    )
    predictions = iso.fit_predict(df_result[numeric_cols].fillna(df_result[numeric_cols].median()))
    is_inlier = (predictions == 1)

    if drop:
        df_result = df_result[is_inlier].reset_index(drop=True)

    return df_result, pd.Series(is_inlier, name="is_inlier")


def build_preprocessing_pipeline(
    numeric_features: List[str],
    categorical_features: List[str],
    ordinal_features: Optional[Dict[str, List[Any]]] = None,
) -> ColumnTransformer:
    """
    Membangun Pipeline ColumnTransformer scikit-learn terintegrasi:
    - Numeric: StandardScaler
    - Categorical (Nominal): OneHotEncoder (handle_unknown='ignore')
    - Ordinal: OrdinalEncoder dengan urutan hierarki yang ditentukan.
    """
    transformers = []

    if numeric_features:
        num_pipeline = Pipeline([
            ("imputer", KNNImputer(n_neighbors=5)),
            ("scaler", StandardScaler()),
        ])
        transformers.append(("num", num_pipeline, numeric_features))

    if categorical_features:
        cat_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])
        transformers.append(("cat", cat_pipeline, categorical_features))

    if ordinal_features:
        for col, categories in ordinal_features.items():
            ord_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("ordinal", OrdinalEncoder(categories=[categories])),
            ])
            transformers.append((f"ord_{col}", ord_pipeline, [col]))

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    return preprocessor


def evaluate_imbalanced_classification(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Evaluasi model pada kondisi class imbalanced.
    Memperingatkan mahasiswa bahwa Akurasi dapat menipu (Accuracy Paradox).
    """
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="weighted")
    prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0)

    return {
        "accuracy": acc,
        "f1_weighted": f1,
        "precision_weighted": prec,
        "recall_weighted": rec,
        "classification_report": report,
        "note": "Perhatikan F1-Score dan Recall untuk kelas minoritas ketimbang mengandalkan Accuracy semata!"
    }
