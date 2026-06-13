import os
import json
import logging
import time
import re
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

print("=" * 60)
print("  SECTION 10: DEPLOYMENT & MONITORING")
print("=" * 60)

# ============================================================
# 1. SETUP LOGGING
# ============================================================
print("\n[1] Setting up logging system...")

os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.FileHandler("logs/predictions.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("sentiment_monitor")

# ============================================================
# 2. MODEL REGISTRY
# ============================================================
print("\n[2] Model Registry:")
print("    Scanning available models in models/ folder...")

available_models = {}
model_dir = "models"

for f in os.listdir(model_dir):
    if f.endswith('.pkl') and 'model' in f:
        path = os.path.join(model_dir, f)
        size_kb = os.path.getsize(path) / 1024
        available_models[f] = {
            'path': path,
            'size_kb': round(size_kb, 1),
            'modified': datetime.fromtimestamp(os.path.getmtime(path)).strftime('%Y-%m-%d %H:%M')
        }
        print(f"    - {f} ({size_kb:.1f} KB, modified: {available_models[f]['modified']})")

# Load metadata
try:
    with open("models/model_metadata.json") as f:
        metadata = json.load(f)
    print(f"\n    Best Model: {metadata['model_name']}")
    print(f"    Accuracy: {metadata['accuracy']*100:.2f}%")
    print(f"    CV Mean: {metadata['cv_mean']*100:.2f}% (+/- {metadata['cv_std']*100:.2f}%)")
    print(f"    Best Params: {metadata['best_params']}")
except FileNotFoundError:
    print("    WARNING: model_metadata.json not found!")
    metadata = None

# ============================================================
# 3. PREDICTION MONITORING CLASS
# ============================================================
print("\n[3] Initializing Prediction Monitor...")

SLANG_MAP = {
    "gak": "tidak", "ga": "tidak", "gk": "tidak", "nggak": "tidak",
    "udah": "sudah", "bgt": "banget", "blm": "belum",
    "klo": "kalau", "krn": "karena", "tp": "tapi", "jd": "jadi",
    "sy": "saya", "gw": "saya", "lu": "anda", "yg": "yang",
}
CUSTOM_STOPWORDS = {'nya', 'si', 'nih', 'tuh', 'dong', 'deh', 'lah', 'kan', 'kok', 'sih', 'aja', 'doang', 'ya'}
NEGATION_WORDS = {
    'tidak', 'bukan', 'gak', 'ga', 'gk', 'nggak', 'ngga', 'kagak',
    'belum', 'blm', 'jangan', 'jgn', 'tak', 'tida', 'enggak',
}

def handle_negation(text):
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
    return ' '.join(result)


class SentimentMonitor:
    def __init__(self, model_path="models/best_model.pkl", vectorizer_path="models/tfidf_vectorizer_improved.pkl"):
        self.model = joblib.load(model_path)
        self.tfidf = joblib.load(vectorizer_path)
        self.stemmer = joblib.load("models/stemmer.pkl")
        self.stopword_remover = joblib.load("models/stopword_remover.pkl")
        self.prediction_log = []

    def clean_text(self, text):
        if not isinstance(text, str):
            return ""
        text = text.lower()
        text = re.sub(r'@\w+|#\w+|http\S+|www\.\S+', '', text)
        text = re.sub(r'[^a-z\s]', '', text)
        words = text.split()
        words = [SLANG_MAP.get(w, w) for w in words]
        text = ' '.join(words)
        text = handle_negation(text)
        text = self.stopword_remover.remove(text)
        words = text.split()
        words = [w for w in words if w not in CUSTOM_STOPWORDS or w.startswith("NOT_")]
        text = ' '.join(words)
        stemmed = []
        for w in text.split():
            if w.startswith("NOT_"):
                stemmed.append("NOT_" + self.stemmer.stem(w[4:]))
            else:
                stemmed.append(self.stemmer.stem(w))
        text = ' '.join(stemmed)
        return re.sub(r'\s+', ' ', text).strip()

    def predict(self, text):
        start_time = time.time()
        cleaned = self.clean_text(text)
        vectorized = self.tfidf.transform([cleaned])
        prediction = self.model.predict(vectorized)[0]
        probabilities = self.model.predict_proba(vectorized)[0]
        confidence = max(probabilities)
        elapsed_ms = (time.time() - start_time) * 1000

        result = {
            'timestamp': datetime.now().isoformat(),
            'text': text[:100],
            'cleaned': cleaned,
            'prediction': prediction,
            'confidence': round(confidence, 4),
            'elapsed_ms': round(elapsed_ms, 2)
        }

        self.prediction_log.append(result)
        logger.info(f"PREDICT | {prediction:10s} | conf={confidence:.4f} | {elapsed_ms:.1f}ms | '{text[:50]}...'")

        return result

    def predict_batch(self, texts):
        results = []
        for text in texts:
            results.append(self.predict(text))
        return results

    def get_stats(self):
        if not self.prediction_log:
            return {"message": "No predictions yet"}

        df = pd.DataFrame(self.prediction_log)
        stats = {
            'total_predictions': len(df),
            'sentiment_distribution': df['prediction'].value_counts().to_dict(),
            'avg_confidence': round(df['confidence'].mean(), 4),
            'min_confidence': round(df['confidence'].min(), 4),
            'avg_latency_ms': round(df['elapsed_ms'].mean(), 2),
            'max_latency_ms': round(df['elapsed_ms'].max(), 2),
        }
        return stats

    def save_log(self, path="logs/prediction_history.csv"):
        if self.prediction_log:
            df = pd.DataFrame(self.prediction_log)
            df.to_csv(path, index=False)
            logger.info(f"Prediction log saved to {path} ({len(df)} entries)")


# ============================================================
# 4. A/B TESTING FRAMEWORK
# ============================================================
print("\n[4] A/B Testing Framework:")

class ABTest:
    def __init__(self, model_a_path, model_b_path, vectorizer_path="models/tfidf_vectorizer_improved.pkl"):
        self.model_a = joblib.load(model_a_path)
        self.model_b = joblib.load(model_b_path)
        self.tfidf = joblib.load(vectorizer_path)
        self.stemmer = joblib.load("models/stemmer.pkl")
        self.stopword_remover = joblib.load("models/stopword_remover.pkl")

    def clean_text(self, text):
        if not isinstance(text, str):
            return ""
        text = text.lower()
        text = re.sub(r'@\w+|#\w+|http\S+|www\.\S+', '', text)
        text = re.sub(r'[^a-z\s]', '', text)
        words = text.split()
        words = [SLANG_MAP.get(w, w) for w in words]
        text = ' '.join(words)
        text = handle_negation(text)
        text = self.stopword_remover.remove(text)
        words = text.split()
        words = [w for w in words if w not in CUSTOM_STOPWORDS or w.startswith("NOT_")]
        text = ' '.join(words)
        stemmed = []
        for w in text.split():
            if w.startswith("NOT_"):
                stemmed.append("NOT_" + self.stemmer.stem(w[4:]))
            else:
                stemmed.append(self.stemmer.stem(w))
        text = ' '.join(stemmed)
        return re.sub(r'\s+', ' ', text).strip()

    def run(self, texts, true_labels=None):
        results = []
        for text in texts:
            cleaned = self.clean_text(text)
            vec = self.tfidf.transform([cleaned])
            pred_a = self.model_a.predict(vec)[0]
            pred_b = self.model_b.predict(vec)[0]
            conf_a = max(self.model_a.predict_proba(vec)[0])
            conf_b = max(self.model_b.predict_proba(vec)[0])

            result = {'text': text[:60], 'model_a': pred_a, 'model_b': pred_b,
                      'conf_a': round(conf_a, 4), 'conf_b': round(conf_b, 4),
                      'agree': pred_a == pred_b}
            if true_labels:
                result['correct_a'] = pred_a == true_labels[len(results)]
                result['correct_b'] = pred_b == true_labels[len(results)]
            results.append(result)
        return pd.DataFrame(results)


# ============================================================
# 5. RUN DEMO
# ============================================================
print("\n[5] Running monitoring demo with 10 test predictions...")

monitor = SentimentMonitor()

test_texts = [
    "Produk ini sangat bagus dan berkualitas tinggi",
    "Pelayanan sangat buruk dan mengecewakan",
    "Barang biasa saja sesuai harga",
    "Saya sangat puas dengan pembelian ini",
    "Penipuan! Barang tidak sesuai gambar",
    "Lumayan oke untuk pemula",
    "Pengiriman cepat dan packaging rapi",
    "Rugi beli di sini, seller tidak amanah",
    "Tidak ada yang istimewa dari produk ini",
    "Best purchase ever! Highly recommended",
]

for text in test_texts:
    monitor.predict(text)

# --- Print Stats ---
stats = monitor.get_stats()
print(f"\n[6] Prediction Statistics:")
print(f"    Total predictions : {stats['total_predictions']}")
print(f"    Sentiment dist    : {stats['sentiment_distribution']}")
print(f"    Avg confidence    : {stats['avg_confidence']:.4f}")
print(f"    Avg latency       : {stats['avg_latency_ms']:.2f} ms")
print(f"    Max latency       : {stats['max_latency_ms']:.2f} ms")

# Save prediction log
monitor.save_log()

# ============================================================
# 6. A/B TEST DEMO
# ============================================================
print("\n[7] A/B Testing Demo...")

model_files = [f for f in os.listdir("models") if f.endswith('.pkl') and 'model' in f and 'vectorizer' not in f]

if len(model_files) >= 2:
    print(f"    Comparing: {model_files[0]} vs {model_files[1]}")
    ab = ABTest(
        os.path.join("models", model_files[0]),
        os.path.join("models", model_files[1])
    )
    ab_results = ab.run(test_texts[:5])
    print(f"\n    Agreement rate: {ab_results['agree'].mean()*100:.1f}%")
    print(f"\n{ab_results.to_string()}")
    ab_results.to_csv("logs/ab_test_results.csv", index=False)
    print(f"\n    A/B test results saved to: logs/ab_test_results.csv")
else:
    print("    Not enough models for A/B test. Need at least 2 models.")

# ============================================================
# 7. DEPLOYMENT CHECKLIST
# ============================================================
print(f"""
{'=' * 60}
  DEPLOYMENT CHECKLIST
{'=' * 60}

  Files yang perlu di-deploy:
  models/
    - best_model.pkl              (model utama)
    - tfidf_vectorizer_improved.pkl (vectorizer)
    - stemmer.pkl                  (stemmer)
    - stopword_remover.pkl         (stopword remover)
    - model_metadata.json          (metadata)
  src/
    - app_improved.py              (Streamlit app)
    - train_improved.py            (training script)
  data/
    - combined_sentiment.csv       (dataset)
    - cleaned_improved.csv         (cleaned data)
  logs/
    - predictions.log              (prediction log)
  requirements.txt

  Command untuk production:

  # Local deployment
  streamlit run src/app_improved.py --server.port 8501

  # Deploy to Streamlit Cloud
  # 1. Push ke GitHub
  # 2. Buka share.streamlit.io
  # 3. Connect repo, set main file: src/app_improved.py

  # Deploy with Docker
  # docker build -t sentiment-app .
  # docker run -p 8501:8501 sentiment-app

  Monitoring:
  - Prediction logs: logs/predictions.log
  - Prediction history: logs/prediction_history.csv
  - A/B test results: logs/ab_test_results.csv
{'=' * 60}
""")

print("  SECTION 10 SELESAI!")
