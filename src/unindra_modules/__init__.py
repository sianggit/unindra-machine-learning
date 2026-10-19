"""
Package unindra_modules: Reusable Machine Learning Modules for Universitas Indraprasta PGRI (Unindra)
Praktikum & UPM/UAS Exam Master Library (Bab 1 s.d. 14).
"""

from .preprocessing import (
    handle_missing_values,
    detect_and_handle_outliers,
    build_preprocessing_pipeline,
    evaluate_imbalanced_classification,
)
from .knn import (
    ManualKNNClassifier,
    evaluate_knn_performance,
    plot_decision_boundaries,
)
from .naive_bayes import (
    ManualGaussianNB,
    ManualMultinomialNB,
    laplace_smoothing_prob,
)
from .decision_tree import (
    calculate_entropy,
    calculate_information_gain,
    calculate_gini,
    calculate_gain_ratio,
    build_decision_tree_classifier,
)
from .kmeans import (
    ManualKMeans1D,
    ManualKMeans2D,
    calculate_wcss,
    elbow_method_analysis,
)
from .svm import (
    calculate_svm_margin,
    evaluate_hyperplane_distance,
)
from .ensemble import (
    calculate_oob_probability,
)
from .dim_reduction import (
    solve_pca_2x2_manual,
)
from .neural_network import (
    neuron_forward_pass_manual,
)
from .deployment import (
    export_model_artifact,
    load_model_artifact,
)

__all__ = [
    "handle_missing_values",
    "detect_and_handle_outliers",
    "build_preprocessing_pipeline",
    "evaluate_imbalanced_classification",
    "ManualKNNClassifier",
    "evaluate_knn_performance",
    "plot_decision_boundaries",
    "ManualGaussianNB",
    "ManualMultinomialNB",
    "laplace_smoothing_prob",
    "calculate_entropy",
    "calculate_information_gain",
    "calculate_gini",
    "calculate_gain_ratio",
    "build_decision_tree_classifier",
    "ManualKMeans1D",
    "ManualKMeans2D",
    "calculate_wcss",
    "elbow_method_analysis",
    "calculate_svm_margin",
    "evaluate_hyperplane_distance",
    "calculate_oob_probability",
    "solve_pca_2x2_manual",
    "neuron_forward_pass_manual",
    "export_model_artifact",
    "load_model_artifact",
]
