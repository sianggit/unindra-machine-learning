"""Model training, evaluation, and inference routines."""

from .train import train_pipeline, get_algorithm_estimator
from .evaluate import evaluate_classification, evaluate_model
from .predict import predict

__all__ = [
    "train_pipeline",
    "get_algorithm_estimator",
    "evaluate_classification",
    "evaluate_model",
    "predict",
]
