import re
import joblib
import pandas as pd
import streamlit as st

# --- Load Model ---
@st.cache_resource
def load_model():
    model = joblib.load("models/sentiment_model.pkl")
    vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
    return model, vectorizer


def clean_text(text):
    """Bersihkan teks dari elemen yang tidak diperlukan."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r'http\S+|www\.\S+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def predict_sentiment(text, model, vectorizer):
    """Prediksi sentimen dari teks."""
    cleaned = clean_text(text)
    vectorized = vectorizer.transform([cleaned])
    prediction = model.predict(vectorized)[0]
    probabilities = model.predict_proba(vectorized)[0]
    classes = model.classes_

    confidence_dict = {}
    for cls, prob in zip(classes, probabilities):
        confidence_dict[cls] = prob

    return prediction, confidence_dict[prediction], confidence_dict


# --- Streamlit App ---
st.set_page_config(page_title="Sentiment Analysis ID", page_icon="📊", layout="centered")

st.title("📊 Indonesian Sentiment Analysis")
st.markdown("Analisis sentimen teks bahasa Indonesia menggunakan Naive Bayes + TF-IDF")

model, tfidf = load_model()

# --- Input Section ---
st.subheader("🔍 Analisis Teks")
user_input = st.text_area(
    "Masukkan teks untuk dianalisis:",
    placeholder="Contoh: Barang ini sangat bagus dan berkualitas!",
    height=100
)

col1, col2 = st.columns([1, 3])
with col1:
    analyze_btn = st.button("🔎 Analisis", type="primary", use_container_width=True)

if analyze_btn and user_input.strip():
    sentiment, confidence, all_probs = predict_sentiment(user_input, model, tfidf)

    sentiment_colors = {
        'positive': '#2ecc71',
        'negative': '#e74c3c',
        'neutral': '#95a5a6'
    }
    sentiment_labels = {
        'positive': '😊 POSITIF',
        'negative': '😠 NEGATIF',
        'neutral': '😐 NETRAL'
    }

    st.markdown("---")
    st.subheader("📋 Hasil Analisis")

    color = sentiment_colors.get(sentiment, '#333333')
    label = sentiment_labels.get(sentiment, sentiment.upper())
    st.markdown(f"**Sentimen:** <span style='color:{color}; font-size:24px; font-weight:bold'>{label}</span>", unsafe_allow_html=True)
    st.markdown(f"**Confidence:** {confidence*100:.2f}%")

    st.markdown("**Detail Probabilitas:**")
    for cls in ['positive', 'negative', 'neutral']:
        prob = all_probs.get(cls, 0)
        st.progress(prob, text=f"{cls}: {prob*100:.1f}%")

    # Save to history
    if 'history' not in st.session_state:
        st.session_state.history = []
    st.session_state.history.append({
        'text': user_input,
        'sentiment': sentiment,
        'confidence': round(confidence, 4)
    })

elif analyze_btn and not user_input.strip():
    st.warning("Silakan masukkan teks terlebih dahulu!")

# --- History Section ---
st.markdown("---")
st.subheader("📜 Riwayat Prediksi")

if 'history' not in st.session_state:
    st.session_state.history = []

if st.session_state.history:
    history_df = pd.DataFrame(st.session_state.history)
    history_df.index = range(1, len(history_df) + 1)
    history_df.index.name = "No"
    st.dataframe(history_df, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("💾 Download Riwayat", use_container_width=True):
            csv = history_df.to_csv(index=False)
            st.download_button(
                "⬇️ Download CSV",
                csv,
                "prediction_history.csv",
                "text/csv",
                use_container_width=True
            )
    with col2:
        if st.button("🗑️ Hapus Riwayat", use_container_width=True):
            st.session_state.history = []
            st.rerun()
else:
    st.info("Belum ada riwayat prediksi. Mulai analisis teks di atas!")

# --- Info Section ---
st.markdown("---")
with st.expander("ℹ️ Tentang Aplikasi"):
    st.markdown("""
    **Aplikasi ini menggunakan:**
    - **Model:** Multinomial Naive Bayes (scikit-learn)
    - **Fitur:** TF-IDF (Term Frequency-Inverse Document Frequency)
    - **Dataset:** 150 data sentimen bahasa Indonesia
    - **Kelas:** Positif, Negatif, Netral

    **Cara kerja:**
    1. Teks dibersihkan (lowercase, hapus URL/mention/simbol)
    2. Teks dikonversi ke vektor angka menggunakan TF-IDF
    3. Model Naive Bayes memprediksi sentimen berdasarkan probabilitas

    **Catatan:** Model ini dilatih dengan dataset kecil (150 data).
    Untuk hasil lebih akurat, gunakan dataset yang lebih besar.
    """)
