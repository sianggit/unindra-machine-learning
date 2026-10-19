"""Dataset loading, validation, and splitting module."""

from pathlib import Path
from typing import Tuple, Optional
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification

from ..utils.logger import setup_logger

logger = setup_logger("ml-workspace.data")

def load_data(file_path: str) -> pd.DataFrame:
    """Loads a dataset from CSV, Excel, or Parquet file format.

    Args:
        file_path: Path to dataset file.

    Returns:
        Loaded pandas DataFrame.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        ValueError: If unsupported file extension is provided.
    """
    path = Path(file_path)
    if not path.exists():
        logger.error("Dataset not found at path: %s", file_path)
        raise FileNotFoundError(f"Dataset file not found at: {file_path}")

    ext = path.suffix.lower()
    if ext == ".csv":
        df = pd.read_csv(path)
    elif ext in [".xls", ".xlsx"]:
        df = pd.read_excel(path)
    elif ext == ".parquet":
        df = pd.read_parquet(path)
    else:
        raise ValueError(f"Unsupported file format '{ext}'. Expected .csv, .xlsx, or .parquet")

    logger.info("Successfully loaded dataset from %s (Shape: %s)", path.name, df.shape)
    return df

def split_dataset(
    df: pd.DataFrame,
    target_column: str,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Splits a DataFrame into train and test feature sets and labels.

    Args:
        df: Input DataFrame.
        target_column: Column name of target label.
        test_size: Proportion of dataset allocated to test split.
        random_state: Deterministic random seed.
        stratify: Whether to preserve label class ratios in splits.

    Returns:
        Tuple of (X_train, X_test, y_train, y_test).
    """
    if target_column not in df.columns:
        raise KeyError(f"Target column '{target_column}' not found in dataframe columns: {list(df.columns)}")

    X = df.drop(columns=[target_column])
    y = df[target_column]

    stratify_target = y if stratify else None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_target
    )

    logger.info(
        "Split completed. Train: %s samples, Test: %s samples (test_size=%.2f)",
        len(X_train), len(X_test), test_size
    )
    return X_train, X_test, y_train, y_test

def generate_sample_dataset(
    n_samples: int = 1000,
    n_features: int = 10,
    n_classes: int = 2,
    random_state: int = 42
) -> pd.DataFrame:
    """Generates a clean synthetic tabular classification dataset for quickstarts.

    Args:
        n_samples: Number of rows to generate.
        n_features: Number of numerical features.
        n_classes: Number of distinct target classes.
        random_state: Deterministic seed.

    Returns:
        DataFrame containing generated features and 'target' column.
    """
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=int(n_features * 0.7),
        n_redundant=int(n_features * 0.2),
        n_classes=n_classes,
        random_state=random_state
    )
    feature_names = [f"feature_{i+1:02d}" for i in range(n_features)]
    df = pd.DataFrame(X, columns=feature_names)
    df["target"] = y
    logger.info("Generated synthetic dataset (Shape: %s, Classes: %s)", df.shape, n_classes)
    return df
