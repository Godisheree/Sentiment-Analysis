import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import joblib
import os

print("=" * 60)
print("  MODEL TRAINING & EVALUATION")
print("  Sentiment Analysis - Indonesian Dataset")
print("=" * 60)

# ============================================================
# STEP 4: FEATURE ENGINEERING (TF-IDF)
# ============================================================
print("\n" + "-" * 60)
print("  STEP 4: FEATURE ENGINEERING (TF-IDF)")
print("-" * 60)

# --- Load Cleaned Data ---
print("\n[4.1] Memuat data bersih...")
df = pd.read_csv("data/cleaned_data.csv")
print(f"      Data dimuat: {len(df)} baris")

# --- TF-IDF Vectorization ---
print("\n[4.2] Mengkonversi teks ke TF-IDF vectors...")
print("      TF-IDF = Term Frequency-Inverse Document Frequency")
print("      Cara kerja: mengubah teks jadi angka berdasarkan")
print("      seberapa penting kata tersebut di dokumen.")

tfidf = TfidfVectorizer(
    max_features=1000,
    ngram_range=(1, 1),
    min_df=1,
    max_df=0.95
)

X = tfidf.fit_transform(df['clean_text'])
y = df['sentiment']

print(f"      Shape TF-IDF matrix: {X.shape}")
print(f"      Jumlah fitur (kata unik): {X.shape[1]}")

# --- Split Data ---
print("\n[4.3] Membagi data: 80% training, 20% testing...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"      Training set: {X_train.shape[0]} data")
print(f"      Testing set : {X_test.shape[0]} data")

# ============================================================
# STEP 5: MODEL TRAINING (NAIVE BAYES)
# ============================================================
print("\n" + "-" * 60)
print("  STEP 5: MODEL TRAINING (NAIVE BAYES)")
print("-" * 60)

print("\n[5.1] Melatih model Naive Bayes...")
print("      Naive Bayes = classifier berbasis probabilitas")
print("      Menggunakan teorema Bayes untuk prediksi.")

model = MultinomialNB(alpha=1.0)
model.fit(X_train, y_train)
print("      Model berhasil dilatih!")

# --- Save Model ---
print("\n[5.2] Menyimpan model dan vectorizer...")
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/sentiment_model.pkl")
joblib.dump(tfidf, "models/tfidf_vectorizer.pkl")
print("      Model disimpan ke: models/sentiment_model.pkl")
print("      Vectorizer disimpan ke: models/tfidf_vectorizer.pkl")

# ============================================================
# STEP 6: MODEL EVALUATION
# ============================================================
print("\n" + "-" * 60)
print("  STEP 6: MODEL EVALUATION")
print("-" * 60)

# --- Prediksi ---
print("\n[6.1] Melakukan prediksi pada data testing...")
y_pred = model.predict(X_test)

# --- Accuracy ---
accuracy = accuracy_score(y_test, y_pred)
print(f"\n[6.2] Hasil Evaluasi:")
print(f"      Accuracy : {accuracy:.4f} ({accuracy*100:.2f}%)")

# --- Precision, Recall, F1 ---
precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

print(f"      Precision: {precision:.4f}")
print(f"      Recall   : {recall:.4f}")
print(f"      F1-Score : {f1:.4f}")

# --- Penjelasan Metrics ---
print("\n[6.3] Penjelasan Metrics:")
print("      Accuracy  = Seberapa sering model benar secara keseluruhan")
print("      Precision = Dari yang diprediksi positif, berapa yang benar positif")
print("      Recall    = Dari yang sebenarnya positif, berapa yang berhasil dideteksi")
print("      F1-Score  = Rata-rata harmonis Precision dan Recall")

# --- Confusion Matrix ---
print("\n[6.4] Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred, labels=['positive', 'negative', 'neutral'])
print(f"                        Prediksi")
print(f"              positive  negative  neutral")
for i, label in enumerate(['positive', 'negative', 'neutral']):
    print(f"  {label:10s}  {cm[i][0]:8d}  {cm[i][1]:8d}  {cm[i][2]:8d}")

# --- Classification Report ---
print("\n[6.5] Classification Report:")
print(classification_report(y_test, y_pred, target_names=['positive', 'negative', 'neutral']))

# --- Training Accuracy (untuk cek overfitting) ---
y_train_pred = model.predict(X_train)
train_accuracy = accuracy_score(y_train, y_train_pred)
print(f"[6.6] Training Accuracy : {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
print(f"      Testing Accuracy  : {accuracy:.4f} ({accuracy*100:.2f}%)")

if train_accuracy - accuracy > 0.1:
    print("      WARNING: Kemungkinan overfitting (gap terlalu besar)")
else:
    print("      OK: Tidak ada overfitting yang signifikan")

print("\n" + "=" * 60)
print("  TRAINING & EVALUATION SELESAI!")
print(f"  Model accuracy: {accuracy*100:.2f}%")
print("=" * 60)
