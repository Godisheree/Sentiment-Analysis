"""Data loading utilities for the Sentiment Analysis project."""

import pandas as pd
from pathlib import Path
from typing import Optional

from src.config import (
    RAW_DATA_PATH,
    COMBINED_DATA_PATH,
    REAL_COMBINED_DATA_PATH,
    CLEANED_DATA_PATH,
)


def load_dataset(path: Optional[Path] = None) -> pd.DataFrame:
    """Load sentiment dataset from CSV.

    Args:
        path: Path to CSV file. Defaults to real_combined_sentiment.csv.

    Returns:
        DataFrame with 'text' and 'sentiment' columns.

    Raises:
        FileNotFoundError: If the dataset file does not exist.
    """
    if path is None:
        for candidate in [REAL_COMBINED_DATA_PATH, COMBINED_DATA_PATH, RAW_DATA_PATH]:
            if candidate.exists():
                path = candidate
                break
        if path is None:
            raise FileNotFoundError(
                f"No dataset found. Searched: {REAL_COMBINED_DATA_PATH}, "
                f"{COMBINED_DATA_PATH}, {RAW_DATA_PATH}"
            )

    df = pd.read_csv(path)
    required_cols = {"text", "sentiment"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Dataset must have columns: {required_cols}")

    return df


def load_cleaned_dataset(path: Optional[Path] = None) -> pd.DataFrame:
    """Load pre-cleaned dataset with 'clean_text' column.

    Args:
        path: Path to cleaned CSV. Defaults to cleaned_improved.csv.

    Returns:
        DataFrame with 'text', 'clean_text', and 'sentiment' columns.
    """
    if path is None:
        path = CLEANED_DATA_PATH
    if not path.exists():
        raise FileNotFoundError(f"Cleaned dataset not found at: {path}")
    return pd.read_csv(path)


def get_class_distribution(df: pd.DataFrame) -> dict[str, int]:
    """Get the distribution of sentiment classes.

    Args:
        df: DataFrame with 'sentiment' column.

    Returns:
        Dictionary mapping sentiment label to count.
    """
    return df["sentiment"].value_counts().to_dict()
