"""Tests for model training utilities."""

import pytest
import numpy as np
from scipy.sparse import csr_matrix

from src.models.traditional import get_model_registry, get_param_grids, create_neural_network
from src.evaluation.metrics import compute_metrics, build_comparison_dataframe


class TestModelRegistry:
    def test_all_models_present(self):
        registry = get_model_registry()
        expected = {"Naive Bayes", "Logistic Regression", "SVM (LinearSVC)", "Random Forest", "XGBoost"}
        assert set(registry.keys()) == expected

    def test_models_are_estimators(self):
        registry = get_model_registry()
        for name, model in registry.items():
            assert hasattr(model, "fit"), f"{name} missing fit method"
            assert hasattr(model, "predict"), f"{name} missing predict method"


class TestParamGrids:
    def test_grids_match_models(self):
        registry = get_model_registry()
        grids = get_param_grids()
        assert set(grids.keys()) == set(registry.keys())

    def test_grids_non_empty(self):
        grids = get_param_grids()
        for name, grid in grids.items():
            assert len(grid) > 0, f"{name} has empty param grid"


class TestNeuralNetwork:
    def test_create_nn(self):
        nn = create_neural_network()
        assert hasattr(nn, "fit")
        assert hasattr(nn, "predict")


class TestComputeMetrics:
    def test_perfect_prediction(self):
        y_true = np.array(["positive", "negative", "neutral", "positive"])
        y_pred = np.array(["positive", "negative", "neutral", "positive"])
        metrics = compute_metrics(y_true, y_pred)
        assert metrics["accuracy"] == 1.0
        assert metrics["f1_score"] == 1.0

    def test_all_wrong(self):
        y_true = np.array(["positive", "positive", "positive"])
        y_pred = np.array(["negative", "negative", "negative"])
        metrics = compute_metrics(y_true, y_pred)
        assert metrics["accuracy"] == 0.0

    def test_partial_correct(self):
        y_true = np.array(["positive", "negative", "neutral", "positive"])
        y_pred = np.array(["positive", "negative", "positive", "negative"])
        metrics = compute_metrics(y_true, y_pred)
        assert metrics["accuracy"] == 0.5
        assert 0 <= metrics["f1_score"] <= 1.0


class TestComparisonDataFrame:
    def test_build_dataframe(self):
        results = {
            "Model A": {"accuracy": 0.9, "precision": 0.88, "recall": 0.9, "f1_score": 0.89},
            "Model B": {"accuracy": 0.85, "precision": 0.84, "recall": 0.85, "f1_score": 0.84},
        }
        df = build_comparison_dataframe(results)
        assert len(df) == 2
        assert "Model" in df.columns
        assert "Accuracy" in df.columns
        assert "F1_Score" in df.columns
