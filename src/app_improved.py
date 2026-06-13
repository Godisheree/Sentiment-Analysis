import re
import json
import joblib
import pandas as pd
import streamlit as st
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

# ============================================================
# HELPER FUNCTIONS
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


@st.cache_resource
def load_resources():
    model = joblib.load("models/best_model.pkl")
    tfidf = joblib.load("models/tfidf_vectorizer_improved.pkl")
    stemmer = joblib.load("models/stemmer.pkl")
    stopword_remover = joblib.load("models/stopword_remover.pkl")
    with open("models/model_metadata.json") as f:
        metadata = json.load(f)
    return model, tfidf, stemmer, stopword_remover, metadata


def advanced_clean(text, stemmer, stopword_remover):
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


def predict_sentiment(text, model, tfidf, stemmer, stopword_remover, threshold=0.5):
    cleaned = advanced_clean(text, stemmer, stopword_remover)
    vectorized = tfidf.transform([cleaned])

    if vectorized.nnz == 0:
        return "uncertain", 0.0, {'positive': 0.33, 'negative': 0.33, 'neutral': 0.34}, cleaned

    prediction = model.predict(vectorized)[0]
    probabilities = model.predict_proba(vectorized)[0]
    classes = model.classes_

    prob_dict = {cls: prob for cls, prob in zip(classes, probabilities)}
    confidence = prob_dict[prediction]

    if confidence < threshold:
        return "uncertain", confidence, prob_dict, cleaned
    return prediction, confidence, prob_dict, cleaned


# ============================================================
# STREAMLIT APP
# ============================================================

st.set_page_config(
    page_title="Sentiment Analysis ID - Improved",
    page_icon="📊",
    layout="wide"
)

model, tfidf, stemmer, stopword_remover, metadata = load_resources()

# --- Header ---
st.title("📊 Indonesian Sentiment Analysis")
st.markdown(f"**Model:** {metadata['model_name']} | **Accuracy:** {metadata['accuracy']*100:.2f}% | **CV Mean:** {metadata['cv_mean']*100:.2f}% (+/- {metadata['cv_std']*100:.2f}%)")

# --- Sidebar ---
with st.sidebar:
    st.header("⚙️ Settings")
    confidence_threshold = st.slider("Confidence Threshold", 0.1, 0.95, 0.5, 0.05,
                                     help="Minimum confidence untuk prediksi valid. Di bawah ini = 'uncertain'")
    st.markdown("---")
    st.subheader("📊 Model Info")
    st.metric("Accuracy", f"{metadata['accuracy']*100:.2f}%")
    st.metric("F1-Score", f"{metadata['f1_score']:.4f}")
    st.metric("CV Mean", f"{metadata['cv_mean']*100:.2f}%")
    st.metric("CV Std", f"{metadata['cv_std']*100:.2f}%")
    st.metric("Training Data", f"{metadata['n_samples']} samples")
    st.metric("Features", f"{metadata['n_features']} terms")

# --- Main Content Tabs ---
tab1, tab2, tab3, tab4 = st.tabs(["🔍 Analisis", "📊 Model Comparison", "📜 Riwayat", "ℹ️ Tentang"])

# ============================================================
# TAB 1: ANALISIS
# ============================================================
with tab1:
    st.subheader("🔍 Analisis Sentimen Teks")

    user_input = st.text_area(
        "Masukkan teks untuk dianalisis:",
        placeholder="Contoh: Barang ini sangat bagus dan berkualitas tinggi!",
        height=100
    )

    col1, col2 = st.columns([1, 4])
    with col1:
        analyze_btn = st.button("🔎 Analisis", type="primary", use_container_width=True)

    if analyze_btn and user_input.strip():
        sentiment, confidence, all_probs, cleaned = predict_sentiment(
            user_input, model, tfidf, stemmer, stopword_remover, confidence_threshold
        )

        sentiment_colors = {'positive': '#2ecc71', 'negative': '#e74c3c', 'neutral': '#95a5a6', 'uncertain': '#f39c12'}
        sentiment_labels = {
            'positive': '😊 POSITIF', 'negative': '😠 NEGATIF',
            'neutral': '😐 NETRAL', 'uncertain': '❓ TIDAK PASTI'
        }

        st.markdown("---")
        st.subheader("📋 Hasil Analisis")

        color = sentiment_colors.get(sentiment, '#333333')
        label = sentiment_labels.get(sentiment, sentiment.upper())
        st.markdown(f"**Sentimen:** <span style='color:{color}; font-size:28px; font-weight:bold'>{label}</span>", unsafe_allow_html=True)
        st.markdown(f"**Confidence:** {confidence*100:.2f}%")

        if sentiment == 'uncertain':
            if confidence == 0.0:
                st.error("Teks tidak mengandung kata yang dikenali model. Gunakan teks bahasa Indonesia yang lebih panjang dan deskriptif.")
            else:
                st.warning(f"Confidence di bawah threshold ({confidence_threshold*100:.0f}%). Model tidak yakin dengan prediksinya.")

        # Probability bars
        st.markdown("**Detail Probabilitas:**")
        prob_cols = st.columns(3)
        for idx, cls in enumerate(['positive', 'negative', 'neutral']):
            prob = all_probs.get(cls, 0)
            with prob_cols[idx]:
                st.metric(cls.capitalize(), f"{prob*100:.1f}%")
                st.progress(prob)

        # Preprocessing info
        with st.expander("🔧 Detail Preprocessing"):
            st.markdown(f"**Teks asli:** {user_input}")
            st.markdown(f"**Setelah cleaning:** {cleaned}")

        # Save to history
        if 'history' not in st.session_state:
            st.session_state.history = []
        st.session_state.history.append({
            'text': user_input,
            'cleaned_text': cleaned,
            'sentiment': sentiment,
            'confidence': round(confidence, 4),
            'prob_positive': round(all_probs.get('positive', 0), 4),
            'prob_negative': round(all_probs.get('negative', 0), 4),
            'prob_neutral': round(all_probs.get('neutral', 0), 4),
        })

    elif analyze_btn and not user_input.strip():
        st.warning("Silakan masukkan teks terlebih dahulu!")

    # Batch analysis
    st.markdown("---")
    with st.expander("📝 Batch Analysis (Multiple Teks)"):
        batch_input = st.text_area(
            "Masukkan beberapa teks (satu per baris):",
            placeholder="Baris 1: Barang bagus sekali\nBaris 2: Pelayanan buruk\nBaris 3: Lumayan oke",
            height=150
        )
        if st.button("📊 Analisis Batch") and batch_input.strip():
            lines = [l.strip() for l in batch_input.strip().split('\n') if l.strip()]
            batch_results = []
            for line in lines:
                sent, conf, probs, cleaned = predict_sentiment(line, model, tfidf, stemmer, stopword_remover, confidence_threshold)
                batch_results.append({
                    'Text': line[:80],
                    'Sentiment': sent,
                    'Confidence': f"{conf*100:.1f}%"
                })
            st.dataframe(pd.DataFrame(batch_results), use_container_width=True)

# ============================================================
# TAB 2: MODEL COMPARISON
# ============================================================
with tab2:
    st.subheader("📊 Perbandingan Model")

    try:
        comp_df = pd.read_csv("data/model_comparison.csv")
        comp_df['Accuracy'] = (comp_df['Accuracy'] * 100).round(2)
        comp_df['Precision'] = comp_df['Precision'].round(4)
        comp_df['Recall'] = comp_df['Recall'].round(4)
        comp_df['F1_Score'] = comp_df['F1_Score'].round(4)
        comp_df = comp_df.rename(columns={'F1_Score': 'F1-Score'})

        st.dataframe(comp_df, use_container_width=True)

        st.markdown("### Visualisasi")
        st.image("data/model_comparison_bar.png", caption="Model Comparison Chart", use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            st.image("data/model_comparison_cm.png", caption="Confusion Matrices", use_container_width=True)
        with col2:
            st.image("data/learning_curve.png", caption="Learning Curve", use_container_width=True)

    except FileNotFoundError:
        st.info("Jalankan `python src/train_improved.py` terlebih dahulu untuk generate comparison data.")

# ============================================================
# TAB 3: RIWAYAT
# ============================================================
with tab3:
    st.subheader("📜 Riwayat Prediksi")

    if 'history' not in st.session_state:
        st.session_state.history = []

    if st.session_state.history:
        history_df = pd.DataFrame(st.session_state.history)
        history_df.index = range(1, len(history_df) + 1)
        history_df.index.name = "No"
        st.dataframe(history_df, use_container_width=True)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Prediksi", len(history_df))
        with col2:
            pos_count = len(history_df[history_df['sentiment'] == 'positive'])
            st.metric("Positif", pos_count)
        with col3:
            neg_count = len(history_df[history_df['sentiment'] == 'negative'])
            st.metric("Negatif", neg_count)

        col1, col2 = st.columns(2)
        with col1:
            csv = history_df.to_csv(index=False)
            st.download_button("⬇️ Download Riwayat CSV", csv, "prediction_history.csv", "text/csv", use_container_width=True)
        with col2:
            if st.button("🗑️ Hapus Riwayat", use_container_width=True):
                st.session_state.history = []
                st.rerun()
    else:
        st.info("Belum ada riwayat prediksi. Mulai analisis di tab pertama!")

# ============================================================
# TAB 4: TENTANG
# ============================================================
with tab4:
    st.subheader("ℹ️ Tentang Aplikasi")

    st.markdown(f"""
    ### Model yang Digunakan
    - **Model:** {metadata['model_name']}
    - **Feature:** TF-IDF (Term Frequency-Inverse Document Frequency)
    - **Preprocessing:** Sastrawi Stemming + Stopword Removal + Slang Normalization
    - **Dataset:** {metadata['n_samples']} data sentimen bahasa Indonesia
    - **Kelas:** Positif, Negatif, Netral

    ### Performance Metrics
    | Metric | Value |
    |--------|-------|
    | Accuracy | {metadata['accuracy']*100:.2f}% |
    | F1-Score | {metadata['f1_score']:.4f} |
    | CV Mean (5-Fold) | {metadata['cv_mean']*100:.2f}% |
    | CV Std | {metadata['cv_std']*100:.2f}% |

    ### Pipeline Preprocessing
    1. Lowercase conversion
    2. Remove URL, mention, hashtag, special characters
    3. Slang normalization (e.g., "gak" -> "tidak")
    4. Indonesian stopword removal (Sastrawi)
    5. Custom stopword removal
    6. Sastrawi stemming (e.g., "mencintai" -> "cinta")

    ### Model yang Dibandingkan
    - Naive Bayes (baseline)
    - Logistic Regression (**best model**)
    - SVM (LinearSVC)
    - Random Forest
    - XGBoost
    - Neural Network (MLPClassifier)

    ### Cara Menjalankan
    ```bash
    # Data preparation
    python src/data_prep.py

    # Training pipeline
    python src/train_improved.py

    # Error analysis
    python src/error_analysis.py

    # Advanced models (optional)
    python src/train_advanced.py

    # Run web app
    streamlit run src/app_improved.py
    ```
    """)
