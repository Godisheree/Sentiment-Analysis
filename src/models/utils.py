"""Model saving, loading, and utility functions."""

import json
import joblib
from pathlib import Path
from typing import Any, Optional

from src.config import (
    BEST_MODEL_PATH,
    TFIDF_PATH,
    STEMMER_PATH,
    STOPWORD_REMOVER_PATH,
    METADATA_PATH,
    MODEL_DIR,
)


def save_model(model: Any, path: Path) -> None:
    """Save a model to disk using joblib.

    Args:
        model: Trained model object.
        path: Destination file path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_model(path: Optional[Path] = None) -> Any:
    """Load a model from disk.

    Args:
        path: Path to model file. Defaults to best_model.pkl.

    Returns:
        Loaded model object.
    """
    if path is None:
        path = BEST_MODEL_PATH
    if not path.exists():
        raise FileNotFoundError(f"Model not found at: {path}")
    return joblib.load(path)


def load_tfidf(path: Optional[Path] = None) -> Any:
    """Load TF-IDF vectorizer from disk.

    Args:
        path: Path to vectorizer file.

    Returns:
        Loaded TF-IDF vectorizer.
    """
    if path is None:
        path = TFIDF_PATH
    return load_model(path)


def load_metadata(path: Optional[Path] = None) -> dict:
    """Load model metadata from JSON.

    Args:
        path: Path to metadata JSON.

    Returns:
        Dictionary with model metadata.
    """
    if path is None:
        path = METADATA_PATH
    if not path.exists():
        raise FileNotFoundError(f"Metadata not found at: {path}")
    with open(path) as f:
        return json.load(f)


def save_metadata(metadata: dict, path: Optional[Path] = None) -> None:
    """Save model metadata to JSON.

    Args:
        metadata: Dictionary with model metadata.
        path: Destination file path.
    """
    if path is None:
        path = METADATA_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(metadata, f, indent=2)


def load_preprocessing_tools() -> tuple:
    """Load stemmer and stopword remover.

    Returns:
        Tuple of (stemmer, stopword_remover).
    """
    stemmer = load_model(STEMMER_PATH)
    stopword_remover = load_model(STOPWORD_REMOVER_PATH)
    return stemmer, stopword_remover


def list_available_models() -> list[dict]:
    """List all available .pkl model files in the models directory.

    Returns:
        List of dicts with model file info.
    """
    models = []
    for f in MODEL_DIR.iterdir():
        if f.suffix == ".pkl" and "model" in f.name:
            models.append({
                "name": f.name,
                "path": f,
                "size_kb": round(f.stat().st_size / 1024, 1),
            })
    return models
