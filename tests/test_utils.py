"""Tests for data loading and model utility functions."""

import pytest
import pandas as pd
from pathlib import Path
from unittest.mock import patch, MagicMock
import tempfile
import os

from src.data.loader import load_dataset, load_cleaned_dataset
from src.models.utils import save_metadata, load_metadata


class TestLoadDataset:
    def test_load_with_explicit_path(self, tmp_path):
        csv_path = tmp_path / "test.csv"
        df = pd.DataFrame({"text": ["hello", "world"], "sentiment": ["positive", "negative"]})
        df.to_csv(csv_path, index=False)
        result = load_dataset(csv_path)
        assert len(result) == 2
        assert "text" in result.columns
        assert "sentiment" in result.columns

    def test_load_missing_columns(self, tmp_path):
        csv_path = tmp_path / "bad.csv"
        df = pd.DataFrame({"col_a": [1], "col_b": [2]})
        df.to_csv(csv_path, index=False)
        with pytest.raises(ValueError, match="must have columns"):
            load_dataset(csv_path)

    def test_load_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_dataset(Path("/nonexistent/path.csv"))


class TestLoadCleanedDataset:
    def test_load_cleaned(self, tmp_path):
        csv_path = tmp_path / "clean.csv"
        df = pd.DataFrame({
            "text": ["hello"],
            "clean_text": ["hello"],
            "sentiment": ["positive"],
        })
        df.to_csv(csv_path, index=False)
        result = load_cleaned_dataset(csv_path)
        assert "clean_text" in result.columns

    def test_load_cleaned_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_cleaned_dataset(Path("/nonexistent.csv"))


class TestMetadata:
    def test_save_and_load_metadata(self, tmp_path):
        meta_path = tmp_path / "meta.json"
        metadata = {
            "model_name": "Test",
            "accuracy": 0.95,
            "f1_score": 0.94,
        }
        save_metadata(metadata, meta_path)
        loaded = load_metadata(meta_path)
        assert loaded["model_name"] == "Test"
        assert loaded["accuracy"] == 0.95

    def test_load_metadata_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_metadata(Path("/nonexistent.json"))
