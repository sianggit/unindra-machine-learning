"""Model inference and prediction service."""

from pathlib import Path
from typing import Union, List, Dict, Any
import joblib
import pandas as pd
import numpy as np

from ..utils.logger import setup_logger

logger = setup_logger("ml-workspace.predict")

def load_model(model_path: str = "models/latest_model.joblib") -> Any:
    """Loads a serialized joblib pipeline model.

    Args:
        model_path: Path to serialized model file.

    Returns:
        Fitted Scikit-learn Pipeline instance.
    """
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model checkpoint not found at: {model_path}")
    model = joblib.load(path)
    logger.info("Successfully loaded model from %s", path.name)
    return model

def predict(
    data: Union[pd.DataFrame, Dict[str, Any], List[Dict[str, Any]]],
    model_path: str = "models/latest_model.joblib",
    return_proba: bool = False
) -> Dict[str, Any]:
    """Generates predictions for input data.

    Args:
        data: DataFrame or dictionary of input features.
        model_path: Path to trained pipeline file.
        return_proba: Whether to include predicted probabilities.

    Returns:
        Dictionary with predictions and optional probabilities.
    """
    model = load_model(model_path)

    if isinstance(data, dict):
        df = pd.DataFrame([data])
    elif isinstance(data, list):
        df = pd.DataFrame(data)
    else:
        df = data

    preds = model.predict(df)
    result = {"predictions": preds.tolist()}

    if return_proba and hasattr(model, "predict_proba"):
        probs = model.predict_proba(df)
        result["probabilities"] = probs.tolist()

    return result
