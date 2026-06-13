"""Tests for the inference/prediction pipeline."""

import pytest
from unittest.mock import patch, MagicMock
import numpy as np
from scipy.sparse import csr_matrix

from src.config import CONFIDENCE_THRESHOLD


class TestSentimentPredictor:
    """Tests for SentimentPredictor using mocked model components."""

    @pytest.fixture
    def mock_predictor(self):
        """Create a mock predictor without loading real model files."""
        with patch("src.inference.joblib") as mock_joblib:
            mock_model = MagicMock()
            mock_model.classes_ = np.array(["negative", "neutral", "positive"])
            mock_model.predict.return_value = np.array(["positive"])
            mock_model.predict_proba.return_value = np.array([[0.1, 0.2, 0.7]])

            mock_tfidf = MagicMock()
            mock_tfidf.transform.return_value = csr_matrix([[1, 2, 3]])

            mock_stemmer = MagicMock()
            mock_stemmer.stem.side_effect = lambda w: w

            mock_stopword = MagicMock()
            mock_stopword.remove.side_effect = lambda t: t

            mock_joblib.load.side_effect = [mock_model, mock_tfidf, mock_stemmer, mock_stopword]

            from src.inference import SentimentPredictor
            predictor = SentimentPredictor()
            return predictor

    def test_predict_returns_dict(self, mock_predictor):
        result = mock_predictor.predict("barang bagus")
        assert isinstance(result, dict)
        assert "sentiment" in result
        assert "confidence" in result
        assert "probabilities" in result

    def test_predict_sentiment_value(self, mock_predictor):
        result = mock_predictor.predict("barang bagus")
        assert result["sentiment"] == "positive"

    def test_predict_confidence_range(self, mock_predictor):
        result = mock_predictor.predict("barang bagus")
        assert 0.0 <= result["confidence"] <= 1.0

    def test_predict_empty_text(self, mock_predictor):
        empty_vec = MagicMock()
        empty_vec.nnz = 0
        mock_predictor.tfidf.transform.return_value = empty_vec
        result = mock_predictor.predict("")
        assert result["sentiment"] == "uncertain"

    def test_predict_batch(self, mock_predictor):
        texts = ["bagus", "jelek", "biasa"]
        results = mock_predictor.predict_batch(texts)
        assert len(results) == 3
        assert all(isinstance(r, dict) for r in results)

    def test_low_confidence_returns_uncertain(self, mock_predictor):
        mock_predictor.model.predict_proba.return_value = np.array([[0.34, 0.33, 0.33]])
        mock_predictor.model.predict.return_value = np.array(["negative"])
        result = mock_predictor.predict("teks ambigu", threshold=0.5)
        assert result["sentiment"] == "uncertain"
