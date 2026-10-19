"""Feature engineering and ColumnTransformer pipeline constructor."""

from typing import List, Tuple, Dict, Any, Optional
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, OneHotEncoder, OrdinalEncoder

from ..utils.logger import setup_logger

logger = setup_logger("ml-workspace.features")

def get_feature_types(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """Identifies numerical and categorical column names automatically.

    Args:
        df: Input feature DataFrame.

    Returns:
        Tuple of (numerical_columns, categorical_columns).
    """
    numerical_cols = list(df.select_dtypes(include=[np.number]).columns)
    categorical_cols = list(df.select_dtypes(include=["object", "category", "bool"]).columns)
    logger.debug("Identified %d numerical and %d categorical features", len(numerical_cols), len(categorical_cols))
    return numerical_cols, categorical_cols

def build_preprocessor_pipeline(
    numerical_cols: List[str],
    categorical_cols: List[str],
    scaling: str = "standard",
    num_impute: str = "median",
    cat_impute: str = "most_frequent",
    encoding: str = "onehot"
) -> ColumnTransformer:
    """Builds a leakage-proof Scikit-Learn ColumnTransformer pipeline.

    Args:
        numerical_cols: List of numerical column names.
        categorical_cols: List of categorical column names.
        scaling: 'standard', 'minmax', 'robust', or 'none'.
        num_impute: 'mean', 'median', or 'most_frequent'.
        cat_impute: 'most_frequent' or 'constant'.
        encoding: 'onehot' or 'ordinal'.

    Returns:
        Configured ColumnTransformer instance.
    """
    transformers = []

    # 1. Numerical Pipeline
    if numerical_cols:
        num_steps = [("imputer", SimpleImputer(strategy=num_impute))]
        if scaling == "standard":
            num_steps.append(("scaler", StandardScaler()))
        elif scaling == "minmax":
            num_steps.append(("scaler", MinMaxScaler()))
        elif scaling == "robust":
            num_steps.append(("scaler", RobustScaler()))

        num_pipeline = Pipeline(steps=num_steps)
        transformers.append(("num", num_pipeline, numerical_cols))

    # 2. Categorical Pipeline
    if categorical_cols:
        cat_steps = [("imputer", SimpleImputer(strategy=cat_impute, fill_value="Missing"))]
        if encoding == "onehot":
            cat_steps.append(("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)))
        elif encoding == "ordinal":
            cat_steps.append(("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)))

        cat_pipeline = Pipeline(steps=cat_steps)
        transformers.append(("cat", cat_pipeline, categorical_cols))

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    logger.info("Built ColumnTransformer with %d feature pipelines (Scaling: %s, Encoding: %s)", len(transformers), scaling, encoding)
    return preprocessor
