"""Traditional ML model definitions and training utilities."""

from typing import Any

import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC

from src.config import RANDOM_STATE


def get_model_registry() -> dict[str, Any]:
    """Return a dictionary of model name to model instance.

    Returns:
        Dictionary mapping model names to sklearn/xgboost estimators.
    """
    return {
        "Naive Bayes": MultinomialNB(alpha=1.0),
        "Logistic Regression": LogisticRegression(
            C=1.0, max_iter=1000, random_state=RANDOM_STATE
        ),
        "SVM (LinearSVC)": LinearSVC(
            C=1.0, max_iter=5000, random_state=RANDOM_STATE, dual="auto"
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1
        ),
        "XGBoost": xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            use_label_encoder=False,
            eval_metric="mlogloss",
            random_state=RANDOM_STATE,
        ),
    }


def get_param_grids() -> dict[str, dict]:
    """Return hyperparameter search grids for each model.

    Returns:
        Dictionary mapping model names to parameter grids.
    """
    return {
        "Naive Bayes": {"alpha": [0.1, 0.5, 1.0, 2.0, 5.0]},
        "Logistic Regression": {
            "C": [0.1, 0.5, 1.0, 5.0, 10.0],
            "max_iter": [500, 1000],
        },
        "SVM (LinearSVC)": {"C": [0.1, 0.5, 1.0, 5.0, 10.0]},
        "Random Forest": {
            "n_estimators": [50, 100, 200],
            "max_depth": [None, 10, 20, 50],
        },
        "XGBoost": {
            "n_estimators": [50, 100, 200],
            "max_depth": [3, 6, 9],
            "learning_rate": [0.01, 0.1, 0.3],
        },
    }


def create_neural_network() -> MLPClassifier:
    """Create a Neural Network classifier (MLPClassifier).

    Returns:
        Configured MLPClassifier instance.
    """
    return MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        solver="adam",
        alpha=0.0001,
        learning_rate="adaptive",
        learning_rate_init=0.001,
        max_iter=100,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=RANDOM_STATE,
        verbose=False,
    )
