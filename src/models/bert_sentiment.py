"""BERT-based sentiment analysis model using HuggingFace Transformers.

Uses IndoBERT or DistilBERT fine-tuned on Indonesian sentiment data.
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from torch.utils.data import Dataset

from src.config import (
    DATA_DIR,
    MODEL_DIR,
    RANDOM_STATE,
    TEST_SIZE,
    SENTIMENT_LABELS,
)

LABEL_MAP = {label: idx for idx, label in enumerate(SENTIMENT_LABELS)}
ID2LABEL = {idx: label for label, idx in LABEL_MAP.items()}

DEFAULT_MODEL_NAME = "indobenchmark/indobert-lite-base-p1"
BERT_MODEL_DIR = MODEL_DIR / "bert_sentiment"


class SentimentDataset(Dataset):
    """PyTorch Dataset for sentiment classification with BERT tokenization."""

    def __init__(self, texts: list[str], labels: list[int], tokenizer, max_length: int = 128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict:
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def load_and_prepare_data(data_path: Optional[Path] = None) -> tuple:
    """Load dataset and split into train/test sets.

    Args:
        data_path: Path to CSV with 'text' and 'sentiment' columns.

    Returns:
        Tuple of (train_texts, train_labels, test_texts, test_labels).
    """
    if data_path is None:
        for candidate in [
            DATA_DIR / "real_combined_sentiment.csv",
            DATA_DIR / "combined_sentiment.csv",
            DATA_DIR / "cleaned_improved.csv",
        ]:
            if candidate.exists():
                data_path = candidate
                break
        if data_path is None:
            raise FileNotFoundError("No dataset found in data/")

    df = pd.read_csv(data_path)

    if "clean_text" in df.columns:
        texts = df["clean_text"].tolist()
    else:
        texts = df["text"].tolist()

    labels = [LABEL_MAP.get(s, 0) for s in df["sentiment"].tolist()]

    return train_test_split(
        texts, labels, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=labels
    )


def train_bert(
    model_name: str = DEFAULT_MODEL_NAME,
    epochs: int = 3,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    max_length: int = 128,
    output_dir: Optional[Path] = None,
) -> dict:
    """Fine-tune a BERT model for sentiment classification.

    Args:
        model_name: HuggingFace model identifier.
        epochs: Number of training epochs.
        batch_size: Training batch size.
        learning_rate: Learning rate.
        max_length: Maximum token sequence length.
        output_dir: Directory to save the trained model.

    Returns:
        Dictionary with training metrics.
    """
    if output_dir is None:
        output_dir = BERT_MODEL_DIR

    print(f"Loading tokenizer and model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=len(SENTIMENT_LABELS)
    )

    print("Preparing data...")
    train_texts, test_texts, train_labels, test_labels = load_and_prepare_data()

    train_dataset = SentimentDataset(train_texts, train_labels, tokenizer, max_length)
    test_dataset = SentimentDataset(test_texts, test_labels, tokenizer, max_length)

    print(f"Training samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")

    training_args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_steps=50,
        report_to="none",
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        acc = accuracy_score(labels, predictions)
        f1 = f1_score(labels, predictions, average="weighted")
        return {"accuracy": acc, "f1": f1}

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )

    print("Starting training...")
    train_result = trainer.train()

    print("Evaluating...")
    eval_result = trainer.evaluate()

    print("Saving model...")
    output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    predictions = trainer.predict(test_dataset)
    y_pred = np.argmax(predictions.predictions, axis=-1)

    print("\nClassification Report:")
    print(classification_report(test_labels, y_pred, target_names=SENTIMENT_LABELS))

    metrics = {
        "model_name": model_name,
        "accuracy": eval_result["eval_accuracy"],
        "f1_score": eval_result["eval_f1"],
        "train_loss": train_result.training_loss,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
    }

    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nModel saved to: {output_dir}")
    print(f"Accuracy: {metrics['accuracy']*100:.2f}%")
    print(f"F1-Score: {metrics['f1_score']:.4f}")

    return metrics


def predict_bert(text: str, model_dir: Optional[Path] = None) -> dict:
    """Run inference with a fine-tuned BERT model.

    Args:
        text: Input text to classify.
        model_dir: Directory containing the saved model.

    Returns:
        Dict with sentiment, confidence, and probabilities.
    """
    if model_dir is None:
        model_dir = BERT_MODEL_DIR

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
    model.eval()

    encoding = tokenizer(
        text, max_length=128, padding="max_length", truncation=True, return_tensors="pt"
    )

    with torch.no_grad():
        outputs = model(**encoding)
        probs = torch.softmax(outputs.logits, dim=-1).squeeze().numpy()

    pred_idx = int(np.argmax(probs))
    return {
        "sentiment": ID2LABEL[pred_idx],
        "confidence": float(probs[pred_idx]),
        "probabilities": {ID2LABEL[i]: float(p) for i, p in enumerate(probs)},
    }


if __name__ == "__main__":
    print("=" * 70)
    print("  BERT SENTIMENT ANALYSIS TRAINING")
    print("=" * 70)

    metrics = train_bert(
        epochs=3,
        batch_size=16,
        learning_rate=2e-5,
    )

    print("\n" + "=" * 70)
    print("  TRAINING COMPLETE")
    print("=" * 70)

    test_texts = [
        "Barang ini sangat bagus dan berkualitas tinggi",
        "Pelayanan sangat buruk dan mengecewakan",
        "Biasa saja sesuai harga",
    ]

    print("\nTest predictions:")
    for text in test_texts:
        result = predict_bert(text)
        print(f"  '{text}' -> {result['sentiment']} ({result['confidence']*100:.1f}%)")
