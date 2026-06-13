"""Text preprocessing pipeline for Indonesian sentiment analysis."""

import re
from typing import Optional

from src.config import SLANG_MAP, CUSTOM_STOPWORDS, NEGATION_WORDS


def handle_negation(text: str) -> str:
    """Prefix words following negation words with NOT_.

    Args:
        text: Input text.

    Returns:
        Text with negated words prefixed NOT_.
    """
    words = text.split()
    result = []
    negate_next = 0
    for word in words:
        if negate_next > 0:
            result.append(f"NOT_{word}")
            negate_next -= 1
        else:
            result.append(word)
        if word in NEGATION_WORDS:
            negate_next = 2
    return " ".join(result)


def normalize_slang(text: str) -> str:
    """Normalize Indonesian slang words to standard form.

    Args:
        text: Input text containing slang.

    Returns:
        Text with slang normalized.
    """
    words = text.split()
    normalized = [SLANG_MAP.get(word, word) for word in words]
    return " ".join(normalized)


def basic_clean(text: str) -> str:
    """Basic text cleaning: lowercase, remove URLs/mentions/symbols.

    Args:
        text: Raw input text.

    Returns:
        Cleaned text.
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#\w+", "", text)
    text = re.sub(r"http\S+|www\.\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def advanced_clean(
    text: str,
    stemmer=None,
    stopword_remover=None,
    use_stemming: bool = True,
    use_stopwords: bool = True,
) -> str:
    """Full preprocessing pipeline for Indonesian text.

    Steps: lowercase -> clean -> normalize slang -> handle negation ->
    stopword removal -> custom stopword removal -> stemming.

    Args:
        text: Raw input text.
        stemmer: Sastrawi stemmer instance (optional).
        stopword_remover: Sastrawi stopword remover instance (optional).
        use_stemming: Whether to apply stemming.
        use_stopwords: Whether to apply stopword removal.

    Returns:
        Fully preprocessed text.
    """
    text = basic_clean(text)
    text = normalize_slang(text)
    text = handle_negation(text)

    if use_stopwords and stopword_remover is not None:
        text = stopword_remover.remove(text)
        words = text.split()
        words = [
            w for w in words
            if w not in CUSTOM_STOPWORDS or w.startswith("NOT_")
        ]
        text = " ".join(words)

    if use_stemming and stemmer is not None:
        words = text.split()
        stemmed = []
        for w in words:
            if w.startswith("NOT_"):
                stemmed.append("NOT_" + stemmer.stem(w[4:]))
            else:
                stemmed.append(stemmer.stem(w))
        text = " ".join(stemmed)

    text = re.sub(r"\s+", " ", text)
    return text.strip()
