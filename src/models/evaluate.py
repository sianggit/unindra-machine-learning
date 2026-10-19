"""Model evaluation and performance metric computation."""

from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report
)

from ..utils.logger import setup_logger
from ..utils.visualization import plot_confusion_matrix

logger = setup_logger("ml-workspace.evaluate")

def evaluate_classification(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """Computes comprehensive classification metrics.

    Args:
        y_true: Ground truth target labels.
        y_pred: Model predictions.
        y_prob: Optional predicted class probability matrix.

    Returns:
        Dictionary containing calculated metrics and report.
    """
    accuracy = float(accuracy_score(y_true, y_pred))
    precision_macro = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    recall_macro = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    f1_macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    metrics = {
        "accuracy": accuracy,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "report": classification_report(y_true, y_pred, zero_division=0, output_dict=True),
        "report_text": classification_report(y_true, y_pred, zero_division=0)
    }

    if y_prob is not None:
        try:
            # Handle binary vs multiclass ROC-AUC
            if y_prob.ndim == 2 and y_prob.shape[1] == 2:
                metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob[:, 1]))
            elif y_prob.ndim == 2 and y_prob.shape[1] > 2:
                metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob, multi_class="ovr"))
        except Exception as e:
            logger.warning("ROC-AUC computation skipped: %s", e)

    logger.info("Evaluation Summary: Accuracy=%.4f, Precision=%.4f, Recall=%.4f, F1=%.4f",
                accuracy, precision_macro, recall_macro, f1_macro)
    return metrics

def evaluate_model(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    output_figure_path: Optional[str] = "reports/figures/confusion_matrix.png"
) -> Dict[str, Any]:
    """Evaluates a fitted model pipeline on a test split, generating plots."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None

    metrics = evaluate_classification(np.array(y_test), y_pred, y_prob)

    if output_figure_path:
        plot_confusion_matrix(
            y_true=np.array(y_test),
            y_pred=y_pred,
            title="Model Evaluation Confusion Matrix",
            output_path=output_figure_path
        )
        logger.info("Confusion matrix figure saved to %s", output_figure_path)

    metrics["predictions"] = y_pred.tolist()
    return metrics
