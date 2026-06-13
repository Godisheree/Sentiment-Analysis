"""Inference pipeline for sentiment prediction."""

import joblib
import numpy as np
from pathlib import Path
from typing import Optional

from src.config import (
    BEST_MODEL_PATH,
    TFIDF_PATH,
    STEMMER_PATH,
    STOPWORD_REMOVER_PATH,
    CONFIDENCE_THRESHOLD,
)
from src.data.preprocessor import advanced_clean


class SentimentPredictor:
    """End-to-end sentiment prediction pipeline.

    Loads model, vectorizer, and preprocessing tools, then provides
    a unified predict method for single or batch inference.
    """

    def __init__(
        self,
        model_path: Optional[Path] = None,
        tfidf_path: Optional[Path] = None,
        stemmer_path: Optional[Path] = None,
        stopword_path: Optional[Path] = None,
    ):
        self.model = joblib.load(model_path or BEST_MODEL_PATH)
        self.tfidf = joblib.load(tfidf_path or TFIDF_PATH)
        self.stemmer = joblib.load(stemmer_path or STEMMER_PATH)
        self.stopword_remover = joblib.load(stopword_path or STOPWORD_REMOVER_PATH)

    def preprocess(self, text: str) -> str:
        """Run full preprocessing pipeline on input text.

        Args:
            text: Raw input text.

        Returns:
            Preprocessed text.
        """
        return advanced_clean(
            text,
            stemmer=self.stemmer,
            stopword_remover=self.stopword_remover,
        )

    def predict(
        self, text: str, threshold: float = CONFIDENCE_THRESHOLD
    ) -> dict:
        """Predict sentiment for a single text.

        Args:
            text: Input text to classify.
            threshold: Minimum confidence for a valid prediction.

        Returns:
            Dict with sentiment, confidence, probabilities, and cleaned_text.
        """
        cleaned = self.preprocess(text)
        vectorized = self.tfidf.transform([cleaned])

        if vectorized.nnz == 0:
            return {
                "sentiment": "uncertain",
                "confidence": 0.0,
                "probabilities": {
                    "positive": 0.33, "negative": 0.33, "neutral": 0.34
                },
                "cleaned_text": cleaned,
            }

        prediction = self.model.predict(vectorized)[0]
        probabilities = self.model.predict_proba(vectorized)[0]
        classes = self.model.classes_

        prob_dict = {cls: float(prob) for cls, prob in zip(classes, probabilities)}
        confidence = prob_dict[prediction]

        if confidence < threshold:
            return {
                "sentiment": "uncertain",
                "confidence": confidence,
                "probabilities": prob_dict,
                "cleaned_text": cleaned,
            }

        return {
            "sentiment": prediction,
            "confidence": confidence,
            "probabilities": prob_dict,
            "cleaned_text": cleaned,
        }

    def predict_batch(
        self, texts: list[str], threshold: float = CONFIDENCE_THRESHOLD
    ) -> list[dict]:
        """Predict sentiment for multiple texts.

        Args:
            texts: List of input texts.
            threshold: Minimum confidence for a valid prediction.

        Returns:
            List of prediction result dicts.
        """
        return [self.predict(text, threshold) for text in texts]
