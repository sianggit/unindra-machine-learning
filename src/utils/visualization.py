"""Publication and academic report visualization utilities."""

from pathlib import Path
from typing import Optional, List
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

# Set publication style
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["font.size"] = 10

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Optional[List[str]] = None,
    title: str = "Confusion Matrix",
    output_path: Optional[str] = "reports/figures/confusion_matrix.png",
    normalize: Optional[str] = None
) -> plt.Figure:
    """Generates and saves a clean, styled confusion matrix heatmap.

    Args:
        y_true: Ground truth target labels.
        y_pred: Predicted class labels.
        labels: Optional class label names for axis ticks.
        title: Plot title.
        output_path: Optional path to save image file.
        normalize: 'true', 'pred', 'all', or None.

    Returns:
        matplotlib Figure object.
    """
    cm = confusion_matrix(y_true, y_pred, normalize=normalize)
    fmt = ".2f" if normalize else "d"

    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        cbar=True,
        xticklabels=labels if labels is not None else "auto",
        yticklabels=labels if labels is not None else "auto",
        ax=ax
    )
    ax.set_title(title, pad=12, fontweight="bold")
    ax.set_xlabel("Predicted Label", labelpad=8)
    ax.set_ylabel("True Label", labelpad=8)
    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight")

    return fig

def plot_correlation_matrix(
    df: pd.DataFrame,
    title: str = "Feature Correlation Matrix",
    output_path: Optional[str] = "reports/figures/correlation_matrix.png"
) -> plt.Figure:
    """Plots and saves an annotated correlation matrix heatmap."""
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()

    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        square=True,
        linewidths=0.5,
        ax=ax
    )
    ax.set_title(title, pad=12, fontweight="bold")
    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight")

    return fig

def plot_distributions(
    df: pd.DataFrame,
    columns: List[str],
    output_path: Optional[str] = "reports/figures/feature_distributions.png"
) -> plt.Figure:
    """Plots univariate feature histograms with KDE curves."""
    n_cols = len(columns)
    fig, axes = plt.subplots(nrows=(n_cols + 1) // 2, ncols=2, figsize=(10, 4 * ((n_cols + 1) // 2)), dpi=300)
    axes = np.array(axes).reshape(-1)

    for i, col in enumerate(columns):
        sns.histplot(df[col], kde=True, ax=axes[i], color="#2b5c8f")
        axes[i].set_title(f"Distribution of {col}", fontweight="bold")

    # Hide unused subplots
    for j in range(n_cols, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight")

    return fig
