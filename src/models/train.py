"""Modular machine learning training pipeline."""

from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from ..data.dataset import load_data, split_dataset, generate_sample_dataset
from ..features.engineer import get_feature_types, build_preprocessor_pipeline
from ..utils.config import load_config
from ..utils.logger import setup_logger

logger = setup_logger("ml-workspace.train")

def get_algorithm_estimator(algorithm_name: str, params: Optional[Dict[str, Any]] = None):
    """Instantiates an estimator from algorithm name string.

    Supported algorithms:
        - RandomForestClassifier
        - LogisticRegression
        - DecisionTreeClassifier
        - KNeighborsClassifier
        - GaussianNB
        - SVC
        - GradientBoostingClassifier
    """
    params = params or {}
    algorithms = {
        "RandomForestClassifier": RandomForestClassifier,
        "LogisticRegression": LogisticRegression,
        "DecisionTreeClassifier": DecisionTreeClassifier,
        "KNeighborsClassifier": KNeighborsClassifier,
        "GaussianNB": GaussianNB,
        "SVC": SVC,
        "GradientBoostingClassifier": GradientBoostingClassifier,
    }

    if algorithm_name not in algorithms:
        available = ", ".join(algorithms.keys())
        raise ValueError(f"Unknown algorithm '{algorithm_name}'. Available: {available}")

    return algorithms[algorithm_name](**params)

def train_pipeline(
    config_path: str = "configs/config.yaml",
    dataset_override: Optional[pd.DataFrame] = None
) -> Dict[str, Any]:
    """Executes the complete machine learning training pipeline.

    1. Loads configuration.
    2. Ingests or generates dataset.
    3. Identifies feature schemas & constructs ColumnTransformer.
    4. Evaluates via Stratified K-Fold cross validation.
    5. Fits full production pipeline on training split.
    6. Persists serializable pipeline to models/ directory.

    Returns:
        Dictionary containing trained pipeline, metrics, and artifact paths.
    """
    config = load_config(config_path)
    data_cfg = config.get("data", {})
    prep_cfg = config.get("preprocessing", {})
    model_cfg = config.get("model", {})
    eval_cfg = config.get("evaluation", {})

    random_seed = config.get("project", {}).get("random_seed", 42)

    # 1. Load Data
    raw_file = data_cfg.get("raw_file", "data/raw/dataset.csv")
    target_col = data_cfg.get("target_column", "target")

    if dataset_override is not None:
        df = dataset_override
    elif Path(raw_file).exists():
        df = load_data(raw_file)
    else:
        logger.warning("Dataset file '%s' not found. Generating synthetic dataset for demonstration.", raw_file)
        df = generate_sample_dataset(n_samples=500, n_features=8, random_state=random_seed)
        Path("data/raw").mkdir(parents=True, exist_ok=True)
        df.to_csv(raw_file, index=False)
        logger.info("Saved synthetic starter dataset to %s", raw_file)

    # 2. Split Dataset
    test_size = data_cfg.get("test_size", 0.2)
    stratify = data_cfg.get("stratify", True)
    X_train, X_test, y_train, y_test = split_dataset(
        df,
        target_column=target_col,
        test_size=test_size,
        random_state=random_seed,
        stratify=stratify
    )

    # Save processed splits
    processed_dir = Path(data_cfg.get("processed_dir", "data/processed"))
    processed_dir.mkdir(parents=True, exist_ok=True)
    X_train.to_parquet(processed_dir / "X_train.parquet") if hasattr(X_train, "to_parquet") else X_train.to_csv(processed_dir / "X_train.csv", index=False)
    X_test.to_parquet(processed_dir / "X_test.parquet") if hasattr(X_test, "to_parquet") else X_test.to_csv(processed_dir / "X_test.csv", index=False)
    y_train.to_csv(processed_dir / "y_train.csv", index=False)
    y_test.to_csv(processed_dir / "y_test.csv", index=False)

    # 3. Build Preprocessor & Estimator
    num_cols, cat_cols = get_feature_types(X_train)
    preprocessor = build_preprocessor_pipeline(
        numerical_cols=num_cols,
        categorical_cols=cat_cols,
        scaling=prep_cfg.get("scaling", "standard"),
        num_impute=prep_cfg.get("imputation", {}).get("numerical_strategy", "median"),
        cat_impute=prep_cfg.get("imputation", {}).get("categorical_strategy", "most_frequent"),
        encoding=prep_cfg.get("encoding", "onehot")
    )

    algorithm_name = model_cfg.get("algorithm", "RandomForestClassifier")
    model_params = model_cfg.get("params", {})
    estimator = get_algorithm_estimator(algorithm_name, model_params)

    # 4. Construct End-to-End Pipeline
    full_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", estimator)
    ])

    # 5. Cross-Validation Evaluation
    cv_folds = eval_cfg.get("cv_folds", 5)
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_seed)
    cv_scores = cross_val_score(full_pipeline, X_train, y_train, cv=skf, scoring="accuracy")

    logger.info("Cross-Validation (%d-Fold) Accuracy: %.4f (+/- %.4f)", cv_folds, np.mean(cv_scores), np.std(cv_scores))

    # 6. Fit Final Pipeline on Full Training Split
    logger.info("Fitting complete pipeline on %d training records...", len(X_train))
    full_pipeline.fit(X_train, y_train)

    # 7. Serialize Artifacts
    model_dir = Path(eval_cfg.get("model_dir", "models"))
    model_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_filename = f"{algorithm_name}_{timestamp}.joblib"
    model_path = model_dir / model_filename
    latest_path = model_dir / "latest_model.joblib"

    joblib.dump(full_pipeline, model_path)
    joblib.dump(full_pipeline, latest_path)
    logger.info("Model pipeline saved successfully to %s", model_path)

    return {
        "pipeline": full_pipeline,
        "model_path": str(model_path),
        "cv_scores": cv_scores.tolist(),
        "cv_mean": float(np.mean(cv_scores)),
        "cv_std": float(np.std(cv_scores)),
        "X_test": X_test,
        "y_test": y_test,
        "features": {
            "numerical": num_cols,
            "categorical": cat_cols
        }
    }
