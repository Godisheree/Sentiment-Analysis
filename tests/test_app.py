"""Tests for configuration and data loading utilities."""

import pytest
import pandas as pd
from pathlib import Path
from unittest.mock import patch, mock_open

from src.config import (
    SLANG_MAP,
    CUSTOM_STOPWORDS,
    NEGATION_WORDS,
    SENTIMENT_LABELS,
    TFIDF_MAX_FEATURES,
    RANDOM_STATE,
    BASE_DIR,
    DATA_DIR,
    MODEL_DIR,
)
from src.data.loader import get_class_distribution


class TestConfig:
    def test_slang_map_non_empty(self):
        assert len(SLANG_MAP) > 0

    def test_slang_map_values_are_strings(self):
        for key, value in SLANG_MAP.items():
            assert isinstance(key, str)
            assert isinstance(value, str)

    def test_custom_stopwords_non_empty(self):
        assert len(CUSTOM_STOPWORDS) > 0

    def test_negation_words_non_empty(self):
        assert len(NEGATION_WORDS) > 0

    def test_sentiment_labels(self):
        assert set(SENTIMENT_LABELS) == {"positive", "negative", "neutral"}

    def test_tfidf_max_features(self):
        assert TFIDF_MAX_FEATURES > 0

    def test_random_state(self):
        assert isinstance(RANDOM_STATE, int)

    def test_directories_defined(self):
        assert BASE_DIR is not None
        assert DATA_DIR is not None
        assert MODEL_DIR is not None


class TestGetClassDistribution:
    def test_balanced_distribution(self):
        df = pd.DataFrame({
            "text": ["a", "b", "c", "d", "e", "f"],
            "sentiment": ["positive", "positive", "negative", "negative", "neutral", "neutral"],
        })
        dist = get_class_distribution(df)
        assert dist["positive"] == 2
        assert dist["negative"] == 2
        assert dist["neutral"] == 2

    def test_single_class(self):
        df = pd.DataFrame({
            "text": ["a", "b"],
            "sentiment": ["positive", "positive"],
        })
        dist = get_class_distribution(df)
        assert dist["positive"] == 2
        assert "negative" not in dist

    def test_empty_dataframe(self):
        df = pd.DataFrame({"text": [], "sentiment": []})
        dist = get_class_distribution(df)
        assert len(dist) == 0
