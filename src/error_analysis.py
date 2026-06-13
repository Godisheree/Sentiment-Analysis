import pandas as pd
import numpy as np
import re
import json
import joblib
import warnings
warnings.filterwarnings('ignore')

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

print("=" * 60)
print("  SECTION 7: ERROR ANALYSIS")
print("  Menganalisis prediksi yang salah")
print("=" * 60)

# ============================================================
# 1. LOAD MODEL & DATA
# ============================================================
print("\n[1] Loading model, vectorizer, dan data...")

model = joblib.load("models/best_model.pkl")
tfidf = joblib.load("models/tfidf_vectorizer_improved.pkl")

with open("models/model_metadata.json") as f:
    metadata = json.load(f)

df = pd.read_csv("data/cleaned_improved.csv")
stemmer = joblib.load("models/stemmer.pkl")
stopword_remover = joblib.load("models/stopword_remover.pkl")

print(f"    Model: {metadata['model_name']}")
print(f"    Data: {len(df)} samples")

# ============================================================
# 2. PREPARE PREPROCESSING
# ============================================================
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

# ============================================================
# 3. PREDICT ALL DATA
# ============================================================
print("\n[2] Melakukan prediksi pada seluruh data...")

df['clean_text'] = df['text'].apply(advanced_clean)
X = tfidf.transform(df['clean_text'])
y_true = df['sentiment'].values
y_pred = model.predict(X)

df['predicted'] = y_pred
df['correct'] = y_true == y_pred

# ============================================================
# 4. ANALYSIS OF MISCLASSIFIED EXAMPLES
# ============================================================
print("\n[3] Analisis Prediksi Salah:")

misclassified = df[df['correct'] == False].copy()
total_wrong = len(misclassified)
total = len(df)

print(f"    Total prediksi    : {total}")
print(f"    Prediksi benar    : {total - total_wrong} ({(total-total_wrong)/total*100:.1f}%)")
print(f"    Prediksi salah    : {total_wrong} ({total_wrong/total*100:.1f}%)")

# --- Error types breakdown ---
print(f"\n[4] Jenis Error (Actual -> Predicted):")
error_types = misclassified.groupby(['sentiment', 'predicted']).size().reset_index(name='count')
error_types = error_types.sort_values('count', ascending=False)

for _, row in error_types.iterrows():
    print(f"    {row['sentiment']:12s} -> {row['predicted']:12s} : {row['count']} samples")

# --- Show worst misclassified examples ---
print(f"\n[5] Contoh Prediksi yang Salah (top 20):")
print(f"{'No':>3} {'Actual':>10} {'Predicted':>10} {'Text'}")
print("-" * 80)

for idx, (_, row) in enumerate(misclassified.head(20).iterrows(), 1):
    text_short = row['text'][:50] + "..." if len(row['text']) > 50 else row['text']
    print(f"{idx:3d} {row['sentiment']:>10} {row['predicted']:>10} {text_short}")

# ============================================================
# 5. WORD FREQUENCY IN MISCLASSIFIED
# ============================================================
print(f"\n[6] Kata paling sering muncul di prediksi salah:")

wrong_words = {}
for text in misclassified['clean_text']:
    for word in text.split():
        wrong_words[word] = wrong_words.get(word, 0) + 1

top_wrong_words = sorted(wrong_words.items(), key=lambda x: x[1], reverse=True)[:15]
for word, count in top_wrong_words:
    print(f"    '{word}' : {count} kali")

print(f"\n    Analisis: Kata-kata ini sering muncul di teks yang sulit diklasifikasikan.")
print(f"    Kemungkinan kata-kata ini ambigu (bisa positif atau negatif tergantung konteks).")

# ============================================================
# 6. CONFUSION MATRIX DETAIL
# ============================================================
print(f"\n[7] Detailed Confusion Matrix:")

labels = ['negative', 'neutral', 'positive']
cm = confusion_matrix(y_true, y_pred, labels=labels)

print(f"\n{'':>15} {'Predicted':>35}")
print(f"{'':>15} {'negative':>12} {'neutral':>12} {'positive':>12}")
for i, label in enumerate(labels):
    print(f"  Actual {label:>8}  {cm[i][0]:>12d} {cm[i][1]:>12d} {cm[i][2]:>12d}")

# ============================================================
# 7. VISUALIZATIONS
# ============================================================
print(f"\n[8] Membuat visualisasi error analysis...")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# --- Confusion Matrix Heatmap ---
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=labels, yticklabels=labels)
axes[0].set_title('Confusion Matrix (Full Data)', fontweight='bold')
axes[0].set_xlabel('Predicted')
axes[0].set_ylabel('Actual')

# --- Error Distribution ---
error_by_class = misclassified['sentiment'].value_counts()
correct_by_class = df[df['correct']]['sentiment'].value_counts()

x_pos = np.arange(len(labels))
width = 0.35
axes[1].bar(x_pos - width/2, [correct_by_class.get(l, 0) for l in labels],
            width, label='Correct', color='#2ecc71')
axes[1].bar(x_pos + width/2, [error_by_class.get(l, 0) for l in labels],
            width, label='Wrong', color='#e74c3c')
axes[1].set_xticks(x_pos)
axes[1].set_xticklabels(labels)
axes[1].set_title('Correct vs Wrong per Class', fontweight='bold')
axes[1].legend()

# --- Error Type Distribution ---
if len(error_types) > 0:
    error_labels = [f"{r['sentiment']}->{r['predicted']}" for _, r in error_types.iterrows()]
    error_counts = error_types['count'].values
    axes[2].barh(error_labels, error_counts, color='#e67e22')
    axes[2].set_title('Error Types (Actual->Predicted)', fontweight='bold')
    axes[2].set_xlabel('Count')
    axes[2].invert_yaxis()

plt.tight_layout()
plt.savefig("data/error_analysis.png", dpi=150, bbox_inches='tight')
print("    Saved to: data/error_analysis.png")

# --- Misclassified text length analysis ---
fig, ax = plt.subplots(figsize=(10, 5))
correct_lens = df[df['correct']]['text'].str.len()
wrong_lens = misclassified['text'].str.len()

ax.hist(correct_lens, bins=30, alpha=0.6, label=f'Correct (n={len(correct_lens)})', color='#2ecc71', density=True)
ax.hist(wrong_lens, bins=30, alpha=0.6, label=f'Wrong (n={len(wrong_lens)})', color='#e74c3c', density=True)
ax.set_xlabel('Text Length (chars)')
ax.set_ylabel('Density')
ax.set_title('Text Length Distribution: Correct vs Wrong Predictions')
ax.legend()
plt.tight_layout()
plt.savefig("data/error_text_length.png", dpi=150, bbox_inches='tight')
print("    Saved to: data/error_text_length.png")

# ============================================================
# 8. SAVE ERROR ANALYSIS RESULTS
# ============================================================
print(f"\n[9] Menyimpan hasil error analysis...")

misclassified[['text', 'clean_text', 'sentiment', 'predicted']].to_csv("data/misclassified.csv", index=False)
print(f"    Misclassified samples saved to: data/misclassified.csv")

# ============================================================
# 9. RECOMMENDATIONS
# ============================================================
print(f"\n{'=' * 60}")
print(f"  REKOMENDASI IMPROVEMENT")
print(f"{'=' * 60}")

accuracy = (total - total_wrong) / total * 100
print(f"""
  1. Model sudah cukup baik (accuracy: {accuracy:.1f}%)
  2. Error terutama di kelas '{error_types.iloc[0]['sentiment']}' -> '{error_types.iloc[0]['predicted']}' ({error_types.iloc[0]['count']} kasus)
  3. Kata ambigu: {', '.join([w for w, _ in top_wrong_words[:5]])}
  4. Rekomendasi:
     - Tambah data training untuk kelas yang sering salah
     - Coba feature engineering lain (Word2Vec, FastText)
     - Coba deep learning (LSTM, BERT) untuk context understanding
     - Tambah negation handling ("tidak bagus" = negatif, bukan positif)
""")

print(f"{'=' * 60}")
print(f"  ERROR ANALYSIS SELESAI!")
print(f"{'=' * 60}")
