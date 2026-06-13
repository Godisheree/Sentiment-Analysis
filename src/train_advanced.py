import pandas as pd
import numpy as np
import re
import json
import joblib
import warnings
warnings.filterwarnings('ignore')

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

print("=" * 60)
print("  SECTION 8: ADVANCED MODELS (DEEP LEARNING BASELINE)")
print("  Neural Network menggunakan sklearn MLPClassifier")
print("=" * 60)

print("""
  PERINGATAN:
  - Script ini menggunakan Neural Network (MLPClassifier)
  - Training lebih lama dari model sebelumnya
  - Untuk deep learning sesungguhnya (LSTM/BERT),
    butuh GPU dan library: PyTorch/TensorFlow + HuggingFace
  - Script ini sebagai baseline untuk perbandingan
""")

# ============================================================
# 1. LOAD & PREPROCESS DATA
# ============================================================
print("[1] Loading data dan preprocessing...")

df = pd.read_csv("data/cleaned_improved.csv")
stemmer = joblib.load("models/stemmer.pkl")
stopword_remover = joblib.load("models/stopword_remover.pkl")

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

def advanced_clean(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'@\w+|#\w+|http\S+|www\.\S+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    words = [SLANG_MAP.get(w, w) for w in words]
    text = ' '.join(words)
    text = handle_negation(text)
    text = stopword_remover.remove(text)
    words = text.split()
    words = [w for w in words if w not in CUSTOM_STOPWORDS or w.startswith("NOT_")]
    text = ' '.join(words)
    stemmed = []
    for w in text.split():
        if w.startswith("NOT_"):
            stemmed.append("NOT_" + stemmer.stem(w[4:]))
        else:
            stemmed.append(stemmer.stem(w))
    text = ' '.join(stemmed)
    return re.sub(r'\s+', ' ', text).strip()

df['clean_text'] = df['text'].apply(advanced_clean)

# TF-IDF
tfidf = TfidfVectorizer(max_features=3000, ngram_range=(1, 2), min_df=2, max_df=0.9, sublinear_tf=True)
X = tfidf.fit_transform(df['clean_text'])
y = df['sentiment']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"    Training: {X_train.shape[0]}, Testing: {X_test.shape[0]}")

# ============================================================
# 2. NEURAL NETWORK (MLPClassifier)
# ============================================================
print("\n[2] Training Neural Network (MLPClassifier)...")
print("    Architecture: 2 hidden layers (128, 64 neurons)")
print("    Activation: ReLU")
print("    Max iterations: 100")
print("    (Ini bisa memakan waktu beberapa menit...)")

nn_model = MLPClassifier(
    hidden_layer_sizes=(128, 64),
    activation='relu',
    solver='adam',
    alpha=0.0001,
    learning_rate='adaptive',
    learning_rate_init=0.001,
    max_iter=100,
    early_stopping=True,
    validation_fraction=0.1,
    random_state=42,
    verbose=False
)

nn_model.fit(X_train, y_train)
y_pred = nn_model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

print(f"\n    Neural Network Results:")
print(f"    Accuracy: {acc*100:.2f}%")
print(f"    F1-Score: {f1:.4f}")

# ============================================================
# 3. COMPARE WITH BEST MODEL
# ============================================================
print("\n[3] Perbandingan dengan model terbaik sebelumnya:")

with open("models/model_metadata.json") as f:
    metadata = json.load(f)

print(f"\n{'Model':<25} {'Accuracy':>10} {'F1-Score':>10}")
print("-" * 45)
print(f"{'Logistic Regression':<25} {metadata['accuracy']*100:>9.2f}% {metadata['f1_score']:>10.4f}")
print(f"{'Neural Network (MLP)':<25} {acc*100:>9.2f}% {f1:>10.4f}")
print("-" * 45)

if acc > metadata['accuracy']:
    print(f"\n    Neural Network LEBIH BAIK! Improvement: +{(acc-metadata['accuracy'])*100:.2f}%")
else:
    print(f"\n    Logistic Regression tetap lebih baik untuk dataset ini.")
    print(f"    Neural Network butuh lebih banyak data dan iterasi untuk optimal.")

# ============================================================
# 4. CLASSIFICATION REPORT
# ============================================================
print(f"\n[4] Neural Network Classification Report:")
print(classification_report(y_test, y_pred, target_names=['negative', 'neutral', 'positive']))

# ============================================================
# 5. TRAINING LOSS CURVE
# ============================================================
print("[5] Saving training loss curve...")
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(nn_model.loss_curve_, color='#3498db', linewidth=2)
ax.set_xlabel('Iterations')
ax.set_ylabel('Loss')
ax.set_title('Neural Network Training Loss Curve')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("data/nn_loss_curve.png", dpi=150, bbox_inches='tight')
print("    Saved to: data/nn_loss_curve.png")

# ============================================================
# 6. SAVE NN MODEL
# ============================================================
print("\n[6] Saving Neural Network model...")
joblib.dump(nn_model, "models/nn_model.pkl")
print("    Saved to: models/nn_model.pkl")

# ============================================================
# 7. INFO TENTANG DEEPER MODELS
# ============================================================
print(f"""
{'=' * 60}
  INFO: DEEP LEARNING LEBIH LANJUT
{'=' * 60}

  Untuk hasil lebih baik lagi, bisa coba:

  1. LSTM / GRU (PyTorch/TensorFlow)
     - Bisa menangkap urutan kata (context)
     - Butuh GPU untuk training cepat
     - pip install torch tensorflow

  2. IndoBERT (Pre-trained Indonesian BERT)
     - Model bahasa Indonesia yang sudah di-train
     - State-of-the-art untuk NLP Indonesia
     - pip install transformers
     - from transformers import AutoTokenizer, AutoModelForSequenceClassification
     - model = AutoModelForSequenceClassification.from_pretrained("indobenchmark/indobert-base-p1")

  3. Fine-tuning approach:
     - Load pre-trained model
     - Fine-tune dengan dataset sentimen kita
     - Biasanya accuracy bisa >95% bahkan dengan data sedikit

  Catatan: Deep learning lebih kompleks dan butuh resource lebih.
  Untuk production dengan data terbatas, Logistic Regression
  atau SVM seringkali sudah cukup dan lebih efisien.
{'=' * 60}
""")

print("  SECTION 8 SELESAI!")
