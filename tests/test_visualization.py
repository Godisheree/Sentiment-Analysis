"""Tests for evaluation metrics and visualization."""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import patch

from src.evaluation.metrics import (
    compute_metrics,
    build_comparison_dataframe,
    plot_confusion_matrices,
    plot_model_comparison,
)


class TestPlotConfusionMatrices:
    def test_plot_saves_file(self, tmp_path):
        results = {
            "Model A": {
                "accuracy": 0.9,
                "precision": 0.88,
                "recall": 0.9,
                "f1_score": 0.89,
                "y_pred": np.array(["positive", "negative", "neutral"]),
            },
        }
        y_test = np.array(["positive", "negative", "neutral"])
        output = tmp_path / "cm.png"
        plot_confusion_matrices(results, y_test, output)
        assert output.exists()


class TestPlotModelComparison:
    def test_plot_saves_file(self, tmp_path):
        results = {
            "Model A": {"accuracy": 0.9, "precision": 0.88, "recall": 0.9, "f1_score": 0.89},
            "Model B": {"accuracy": 0.85, "precision": 0.84, "recall": 0.85, "f1_score": 0.84},
        }
        output = tmp_path / "bar.png"
        plot_model_comparison(results, output)
        assert output.exists()
