import re
import joblib
import pandas as pd
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

print("=" * 60)
print("  PREDICT SENTIMENT DARI TEXT BARU")
print("  Sentiment Analysis - Improved Model")
print("=" * 60)

# --- Preprocessing Setup ---
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

stemmer = StemmerFactory().create_stemmer()
stopword_remover = StopWordRemoverFactory().create_stop_word_remover()


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


def clean_text(text):
    """Full preprocessing pipeline (same as training)."""
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


def predict_sentiment(text, model, vectorizer):
    """Prediksi sentimen dari teks baru."""
    cleaned = clean_text(text)
    vectorized = vectorizer.transform([cleaned])

    if vectorized.nnz == 0:
        print("    WARNING: Teks tidak mengandung kata yang dikenali model.")
        print("    Gunakan teks bahasa Indonesia yang lebih panjang.")
        return "uncertain", 0.0, {'positive': 0.33, 'negative': 0.33, 'neutral': 0.34}

    prediction = model.predict(vectorized)[0]
    probabilities = model.predict_proba(vectorized)[0]
    classes = model.classes_

    confidence_dict = {}
    for cls, prob in zip(classes, probabilities):
        confidence_dict[cls] = prob

    confidence = confidence_dict[prediction]
    return prediction, confidence, confidence_dict


# --- Load Model ---
print("\n[1] Memuat model improved...")
model = joblib.load("models/best_model.pkl")
tfidf = joblib.load("models/tfidf_vectorizer_improved.pkl")
print("    Model improved berhasil dimuat!")

# --- Test Cases ---
print("\n[2] Menguji dengan 10 contoh teks (termasuk negasi):")
test_cases = [
    "Barang ini bagus banget, saya sangat puas dengan kualitasnya",
    "Pelayanan sangat buruk, barang rusak dan tidak ada tanggapan",
    "Produk sudah sampai, biasa saja sesuai harga",
    "Sangat kecewa! Barang tidak sesuai gambar sama sekali",
    "Lumayan oke untuk harga segini, tidak ada masalah",
    "TIDAK BAGUS sama sekali, nyesel beli",
    "Bukan produk yang bagus, kualitasnya jelek",
    "Gak recommended, pelayanannya lama banget",
    "Barangnya oke, pengiriman juga cepat",
    "Saya tidak puas dengan produk ini, kualitas buruk",
]

results = []
for i, text in enumerate(test_cases, 1):
    sentiment, confidence, all_probs = predict_sentiment(text, model, tfidf)
    results.append({
        'text': text,
        'predicted_sentiment': sentiment,
        'confidence': round(confidence, 4)
    })

    emoji = {'positive': ':)', 'negative': ':(', 'neutral': ':|', 'uncertain': '?'}
    print(f"\n    --- Test {i} ---")
    print(f"    Teks      : {text}")
    print(f"    Sentimen  : {sentiment} {emoji.get(sentiment, '')}")
    print(f"    Confidence: {confidence*100:.2f}%")
    print(f"    Detail    : ", end="")
    for cls, prob in all_probs.items():
        print(f"{cls}={prob*100:.1f}% ", end="")
    print()

# --- Simpan Hasil ---
results_df = pd.DataFrame(results)
results_df.to_csv("data/predictions.csv", index=False)
print(f"\n[3] Hasil prediksi disimpan ke: data/predictions.csv")

# --- Interactive Mode ---
print("\n" + "=" * 60)
print("  MODE INTERAKTIF")
print("  Ketik teks untuk prediksi (ketik 'quit' untuk keluar)")
print("=" * 60)

while True:
    user_input = input("\n  Masukkan teks: ").strip()
    if user_input.lower() in ('quit', 'exit', 'q', ''):
        break

    sentiment, confidence, all_probs = predict_sentiment(user_input, model, tfidf)
    emoji = {'positive': ':)', 'negative': ':(', 'neutral': ':|', 'uncertain': '?'}
    print(f"  Sentimen   : {sentiment} {emoji.get(sentiment, '')}")
    print(f"  Confidence : {confidence*100:.2f}%")

print("\n  Terima kasih! Program selesai.")
