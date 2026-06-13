"""Evaluation metrics and visualization utilities."""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

from src.config import SENTIMENT_LABELS, DATA_DIR


def compute_metrics(
    y_true: np.ndarray, y_pred: np.ndarray
) -> dict[str, float]:
    """Compute classification metrics.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.

    Returns:
        Dictionary with accuracy, precision, recall, and f1_score.
    """
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1_score": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }


def plot_confusion_matrices(
    results: dict[str, dict],
    y_test: np.ndarray,
    output_path: Optional[Path] = None,
) -> None:
    """Plot confusion matrices for all models.

    Args:
        results: Dict mapping model names to result dicts with 'y_pred'.
        y_test: True test labels.
        output_path: Where to save the plot.
    """
    if output_path is None:
        output_path = DATA_DIR / "model_comparison_cm.png"

    n = len(results)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))
    axes = np.atleast_1d(axes).flatten()

    for idx, (name, metrics) in enumerate(results.items()):
        cm = confusion_matrix(y_test, metrics["y_pred"], labels=SENTIMENT_LABELS)
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues", ax=axes[idx],
            xticklabels=SENTIMENT_LABELS, yticklabels=SENTIMENT_LABELS,
        )
        axes[idx].set_title(
            f"{name}\nAcc: {metrics['accuracy']*100:.1f}%", fontweight="bold"
        )
        axes[idx].set_xlabel("Predicted")
        axes[idx].set_ylabel("Actual")

    for idx in range(n, len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_model_comparison(
    results: dict[str, dict], output_path: Optional[Path] = None
) -> None:
    """Plot bar chart comparing model accuracy and F1-score.

    Args:
        results: Dict mapping model names to result dicts.
        output_path: Where to save the plot.
    """
    if output_path is None:
        output_path = DATA_DIR / "model_comparison_bar.png"

    model_names = list(results.keys())
    accuracies = [results[m]["accuracy"] * 100 for m in model_names]
    f1_scores = [results[m]["f1_score"] * 100 for m in model_names]

    x = np.arange(len(model_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width / 2, accuracies, width, label="Accuracy (%)", color="#3498db")
    bars2 = ax.bar(x + width / 2, f1_scores, width, label="F1-Score (%)", color="#e74c3c")
    ax.set_ylabel("Score (%)")
    ax.set_title("Model Comparison: Accuracy vs F1-Score")
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=15, ha="right")
    ax.legend()
    ax.bar_label(bars1, fmt="%.1f", padding=3)
    ax.bar_label(bars2, fmt="%.1f", padding=3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def build_comparison_dataframe(results: dict[str, dict]) -> pd.DataFrame:
    """Build a comparison DataFrame from model results.

    Args:
        results: Dict mapping model names to result dicts.

    Returns:
        DataFrame with Model, Accuracy, Precision, Recall, F1_Score columns.
    """
    return pd.DataFrame([
        {
            "Model": name,
            "Accuracy": m["accuracy"],
            "Precision": m["precision"],
            "Recall": m["recall"],
            "F1_Score": m["f1_score"],
        }
        for name, m in results.items()
    ])
