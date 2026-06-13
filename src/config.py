"""Central configuration for the Sentiment Analysis project."""

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
LOG_DIR = BASE_DIR / "logs"

# Model paths
BEST_MODEL_PATH = MODEL_DIR / "best_model.pkl"
TFIDF_PATH = MODEL_DIR / "tfidf_vectorizer_improved.pkl"
STEMMER_PATH = MODEL_DIR / "stemmer.pkl"
STOPWORD_REMOVER_PATH = MODEL_DIR / "stopword_remover.pkl"
METADATA_PATH = MODEL_DIR / "model_metadata.json"

# Dataset paths
RAW_DATA_PATH = DATA_DIR / "raw_sentiment.csv"
COMBINED_DATA_PATH = DATA_DIR / "combined_sentiment.csv"
REAL_COMBINED_DATA_PATH = DATA_DIR / "real_combined_sentiment.csv"
CLEANED_DATA_PATH = DATA_DIR / "cleaned_improved.csv"

# Preprocessing constants
SLANG_MAP: dict[str, str] = {
    "gak": "tidak", "ga": "tidak", "gk": "tidak", "enggak": "tidak",
    "nggak": "tidak", "ngga": "tidak", "kagak": "tidak",
    "udah": "sudah", "udh": "sudah", "sdh": "sudah",
    "bgt": "sangat", "bngt": "sangat", "bgtt": "sangat",
    "blm": "belum", "belom": "belum",
    "dgn": "dengan", "dg": "dengan",
    "klo": "kalau", "kalo": "kalau", "kl": "kalau",
    "krn": "karena", "karna": "karena", "krena": "karena",
    "tp": "tapi", "tapi": "tapi", "tpi": "tapi",
    "jd": "jadi", "jdi": "jadi",
    "sy": "saya", "sya": "saya", "syaa": "saya",
    "gw": "saya", "gue": "saya", "gua": "saya", "aku": "saya",
    "lu": "anda", "lo": "anda", "loe": "anda", "elu": "anda",
    "yg": "yang", "g": "yang",
    "dr": "dari", "dri": "dari",
    "bs": "bisa", "bsa": "bisa",
    "jg": "juga", "jga": "juga",
    "thx": "terima kasih", "thnx": "terima kasih", "tq": "terima kasih",
    "ok": "oke", "oke": "oke", "okey": "oke",
    "gt": "begitu", "gitu": "begitu",
    "gpp": "tidak apa apa", "gapapa": "tidak apa apa",
    "bener": "benar", "beneran": "benar",
    "emang": "memang", "emg": "memang",
    "kayak": "seperti", "kya": "seperti",
    "pake": "pakai", "pk": "pakai",
    "bikin": "buat", "bikinin": "buat",
    "mantap": "bagus", "mantul": "bagus",
    "kereeen": "bagus", "baguus": "bagus",
    "jeleek": "jelek", "buruuuk": "buruk",
    "murcee": "murah", "mahaaal": "mahal",
}

CUSTOM_STOPWORDS: set[str] = {
    "nya", "si", "nih", "tuh", "dong", "deh", "lah", "kan", "kok",
    "sih", "nah", "loh", "yuk", "aja", "doang", "ya",
}

NEGATION_WORDS: set[str] = {
    "tidak", "bukan", "gak", "ga", "gk", "nggak", "ngga", "kagak",
    "belum", "blm", "jangan", "jgn", "tak", "tida", "enggak",
}

# TF-IDF parameters
TFIDF_MAX_FEATURES = 3000
TFIDF_NGRAM_RANGE = (1, 2)
TFIDF_MIN_DF = 2
TFIDF_MAX_DF = 0.9
TFIDF_SUBLINEAR_TF = True

# Training parameters
TEST_SIZE = 0.2
RANDOM_STATE = 42
CV_FOLDS = 5

# Sentiment labels
SENTIMENT_LABELS = ["positive", "negative", "neutral"]

# App configuration
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.5"))
MAX_TEXT_LENGTH = int(os.getenv("MAX_TEXT_LENGTH", "500"))
STREAMLIT_PORT = int(os.getenv("STREAMLIT_PORT", "8501"))
DEBUG = os.getenv("DEBUG", "False").lower() == "true"
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
