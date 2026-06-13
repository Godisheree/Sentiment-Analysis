import pandas as pd
import numpy as np
import re
import os
import json
import warnings
warnings.filterwarnings('ignore')

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV, learning_curve
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import xgboost as xgb
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

print("=" * 70)
print("  IMPROVED TRAINING PIPELINE")
print("  Sections 2-6: Preprocessing, Features, Models, Tuning, CV")
print("=" * 70)

# ============================================================
# SECTION 2: ADVANCED DATA PREPROCESSING
# ============================================================
print("\n" + "=" * 70)
print("  SECTION 2: ADVANCED DATA PREPROCESSING")
print("=" * 70)

# --- Slang normalization dictionary ---
SLANG_MAP = {
    "gak": "tidak", "ga": "tidak", "gk": "tidak", "enggak": "tidak",
    "nggak": "tidak", "ngga": "tidak", "kagak": "tidak",
    "udah": "sudah", "udh": "sudah", "sdh": "sudah",
    "bgt": "banget", "bngt": "banget", "bgtt": "banget",
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
    "bisa": "bisa", "bs": "bisa", "bsa": "bisa",
    "jg": "juga", "jga": "juga",
    "thx": "terima kasih", "thnx": "terima kasih", "tq": "terima kasih",
    "ok": "oke", "oke": "oke", "okey": "oke",
    "gt": "begitu", "gitu": "begitu",
    "gpp": "tidak apa apa", "gapapa": "tidak apa apa",
    "bener": "benar", "beneran": "benar",
    "emang": "memang", "emg": "memang",
    "kayak": "seperti", "kya": "seperti", "kyaK": "seperti",
    "banget": "sangat", "bgt": "sangat",
    "pake": "pakai", "pk": "pakai",
    "bikin": "buat", "bikinin": "buat",
    "mantap": "bagus", "mantul": "bagus",
    "kereeen": "bagus", "baguus": "bagus",
    "jeleek": "jelek", "buruuuk": "buruk",
    "murcee": "murah", "mahaaal": "mahal",
}

def normalize_slang(text):
    """Normalisasi kata slang ke bentuk baku."""
    words = text.split()
    normalized = [SLANG_MAP.get(word, word) for word in words]
    return ' '.join(normalized)

def basic_clean(text):
    """Pembersihan dasar: lowercase, hapus URL/mention/simbol."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r'http\S+|www\.\S+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# --- Initialize Sastrawi ---
print("\n[2.1] Initializing Sastrawi stemmer & stopword remover...")
stemmer_factory = StemmerFactory()
stemmer = stemmer_factory.create_stemmer()
stopword_factory = StopWordRemoverFactory()
stopword_remover = stopword_factory.create_stop_word_remover()

# Additional custom stopwords
CUSTOM_STOPWORDS = {
    'nya', 'si', 'nih', 'tuh', 'dong', 'deh', 'lah', 'kan', 'kok',
    'sih', 'nah', 'loh', 'yuk', 'aja', 'doang', 'aja', 'ya', 'nih',
}

NEGATION_WORDS = {
    'tidak', 'bukan', 'gak', 'ga', 'gk', 'nggak', 'ngga', 'kagak',
    'belum', 'blm', 'jangan', 'jgn', 'tak', 'tida', 'enggak',
}

def handle_negation(text):
    """Prefix kata setelah negasi dengan NOT_ agar model bisa bedakan."""
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

def advanced_clean(text, use_stemming=True, use_stopwords=True):
    """Pembersihan teks tingkat lanjut."""
    text = basic_clean(text)
    text = normalize_slang(text)
    text = handle_negation(text)

    if use_stopwords:
        text = stopword_remover.remove(text)
        words = text.split()
        words = [w for w in words if w not in CUSTOM_STOPWORDS or w.startswith("NOT_")]
        text = ' '.join(words)

    if use_stemming:
        words = text.split()
        stemmed = []
        for w in words:
            if w.startswith("NOT_"):
                stemmed.append("NOT_" + stemmer.stem(w[4:]))
            else:
                stemmed.append(stemmer.stem(w))
        text = ' '.join(stemmed)

    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# --- Load Data ---
print("\n[2.2] Loading combined dataset (real + synthetic)...")
df = pd.read_csv("data/real_combined_sentiment.csv")
print(f"      Loaded: {len(df)} samples")

# --- Apply Preprocessing ---
print("\n[2.3] Applying advanced preprocessing...")
print("      - Slang normalization")
print("      - Stopword removal (Indonesian)")
print("      - Sastrawi stemming")

df['clean_text'] = df['text'].apply(lambda x: advanced_clean(x))

# Remove empty after cleaning
empty_count = (df['clean_text'].str.len() == 0).sum()
df = df[df['clean_text'].str.len() > 0].reset_index(drop=True)
print(f"      Removed {empty_count} empty texts")
print(f"      Remaining: {len(df)} samples")

# --- Show before/after comparison ---
print("\n[2.4] Before vs After Preprocessing:")
for i in [0, len(df)//2, len(df)-1]:
    print(f"\n      BEFORE: {df['text'].iloc[i]}")
    print(f"      AFTER : {df['clean_text'].iloc[i]}")

# --- Save cleaned data ---
df[['text', 'clean_text', 'sentiment']].to_csv("data/cleaned_improved.csv", index=False)
print(f"\n[2.5] Cleaned data saved to: data/cleaned_improved.csv")

# ============================================================
# SECTION 3: FEATURE ENGINEERING (TF-IDF)
# ============================================================
print("\n" + "=" * 70)
print("  SECTION 3: FEATURE ENGINEERING")
print("=" * 70)

print("\n[3.1] Creating TF-IDF features...")

tfidf = TfidfVectorizer(
    max_features=3000,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.9,
    sublinear_tf=True
)

X = tfidf.fit_transform(df['clean_text'])
y = df['sentiment']

print(f"      TF-IDF matrix shape: {X.shape}")
print(f"      Features (unique terms): {X.shape[1]}")
print(f"      Parameters: max_features=3000, ngram=(1,2), sublinear_tf=True")

# --- Split Data ---
print("\n[3.2] Splitting data: 80% train, 20% test...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"      Training: {X_train.shape[0]} samples")
print(f"      Testing : {X_test.shape[0]} samples")

# ============================================================
# SECTION 4: MODEL COMPARISON (5 MODELS)
# ============================================================
print("\n" + "=" * 70)
print("  SECTION 4: MODEL COMPARISON")
print("=" * 70)

models = {
    'Naive Bayes': MultinomialNB(alpha=1.0),
    'Logistic Regression': LogisticRegression(C=1.0, max_iter=1000, random_state=42),
    'SVM (LinearSVC)': LinearSVC(C=1.0, max_iter=5000, random_state=42, dual='auto'),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    'XGBoost': xgb.XGBClassifier(
        n_estimators=100, max_depth=6, learning_rate=0.1,
        use_label_encoder=False, eval_metric='mlogloss', random_state=42
    ),
}

results = {}
trained_models = {}

print("\n[4.1] Training and evaluating 5 models...\n")

for name, model in models.items():
    print(f"      Training: {name}...", end=" ")

    if name == 'XGBoost':
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        y_train_enc = le.fit_transform(y_train)
        y_test_enc = le.transform(y_test)
        model.fit(X_train, y_train_enc)
        y_pred_enc = model.predict(X_test)
        y_pred = le.inverse_transform(y_pred_enc)
    elif name == 'SVM (LinearSVC)':
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

    results[name] = {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1_score': f1,
        'y_pred': y_pred
    }
    trained_models[name] = model

    print(f"Accuracy: {acc*100:.2f}% | F1: {f1:.4f}")

# --- Comparison Table ---
print("\n" + "=" * 70)
print("  MODEL COMPARISON TABLE")
print("=" * 70)
print(f"\n{'Model':<25} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1-Score':>10}")
print("-" * 65)

best_model_name = ""
best_f1 = 0

for name, metrics in results.items():
    print(f"{name:<25} {metrics['accuracy']*100:>9.2f}% {metrics['precision']:>10.4f} {metrics['recall']:>10.4f} {metrics['f1_score']:>10.4f}")
    if metrics['f1_score'] > best_f1:
        best_f1 = metrics['f1_score']
        best_model_name = name

print("-" * 65)
print(f"\n  BEST MODEL: {best_model_name} (F1-Score: {best_f1:.4f})")

# --- Confusion Matrices ---
print("\n[4.2] Confusion Matrices:")
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.flatten()
labels = ['negative', 'neutral', 'positive']

for idx, (name, metrics) in enumerate(results.items()):
    cm = confusion_matrix(y_test, metrics['y_pred'], labels=labels)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                xticklabels=labels, yticklabels=labels)
    axes[idx].set_title(f'{name}\nAcc: {metrics["accuracy"]*100:.1f}%', fontweight='bold')
    axes[idx].set_xlabel('Predicted')
    axes[idx].set_ylabel('Actual')

axes[5].set_visible(False)
plt.tight_layout()
plt.savefig("data/model_comparison_cm.png", dpi=150, bbox_inches='tight')
print("    Saved confusion matrices to: data/model_comparison_cm.png")

# --- Bar chart comparison ---
fig, ax = plt.subplots(figsize=(12, 6))
model_names = list(results.keys())
accuracies = [results[m]['accuracy'] * 100 for m in model_names]
f1_scores = [results[m]['f1_score'] for m in model_names]

x = np.arange(len(model_names))
width = 0.35
bars1 = ax.bar(x - width/2, accuracies, width, label='Accuracy (%)', color='#3498db')
bars2 = ax.bar(x + width/2, [f*100 for f in f1_scores], width, label='F1-Score (%)', color='#e74c3c')
ax.set_ylabel('Score (%)')
ax.set_title('Model Comparison: Accuracy vs F1-Score')
ax.set_xticks(x)
ax.set_xticklabels(model_names, rotation=15, ha='right')
ax.legend()
ax.bar_label(bars1, fmt='%.1f', padding=3)
ax.bar_label(bars2, fmt='%.1f', padding=3)
plt.tight_layout()
plt.savefig("data/model_comparison_bar.png", dpi=150, bbox_inches='tight')
print("    Saved bar chart to: data/model_comparison_bar.png")

# ============================================================
# SECTION 5: HYPERPARAMETER TUNING
# ============================================================
print("\n" + "=" * 70)
print("  SECTION 5: HYPERPARAMETER TUNING")
print("=" * 70)

print(f"\n[5.1] Tuning best model: {best_model_name}")
print("      Using GridSearchCV to find optimal hyperparameters...")

# Define param grids for the best model
param_grids = {
    'Naive Bayes': {'alpha': [0.1, 0.5, 1.0, 2.0, 5.0]},
    'Logistic Regression': {'C': [0.1, 0.5, 1.0, 5.0, 10.0], 'max_iter': [500, 1000]},
    'SVM (LinearSVC)': {'C': [0.1, 0.5, 1.0, 5.0, 10.0]},
    'Random Forest': {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20, 50]},
    'XGBoost': {'n_estimators': [50, 100, 200], 'max_depth': [3, 6, 9], 'learning_rate': [0.01, 0.1, 0.3]},
}

best_params = param_grids.get(best_model_name, {})
print(f"      Parameter grid: {best_params}")

# Create a fresh model instance for tuning
if best_model_name == 'XGBoost':
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    base_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
    grid_search = GridSearchCV(base_model, best_params, cv=3, scoring='f1_weighted', n_jobs=-1, verbose=0)
    grid_search.fit(X_train, y_train_enc)
    print(f"      Best parameters: {grid_search.best_params_}")
    print(f"      Best CV F1-score: {grid_search.best_score_:.4f}")

    # Retrain with best params
    tuned_model = xgb.XGBClassifier(
        use_label_encoder=False, eval_metric='mlogloss', random_state=42,
        **grid_search.best_params_
    )
    tuned_model.fit(X_train, y_train_enc)
    y_pred_tuned = le.inverse_transform(tuned_model.predict(X_test))
else:
    base_model = type(trained_models[best_model_name])()
    # Copy base params
    if best_model_name == 'SVM (LinearSVC)':
        base_model = LinearSVC(max_iter=5000, random_state=42, dual='auto')
    elif best_model_name == 'Logistic Regression':
        base_model = LogisticRegression(max_iter=1000, random_state=42)
    elif best_model_name == 'Random Forest':
        base_model = RandomForestClassifier(random_state=42, n_jobs=-1)

    grid_search = GridSearchCV(base_model, best_params, cv=3, scoring='f1_weighted', n_jobs=-1, verbose=0)
    grid_search.fit(X_train, y_train)
    print(f"      Best parameters: {grid_search.best_params_}")
    print(f"      Best CV F1-score: {grid_search.best_score_:.4f}")

    tuned_model = grid_search.best_estimator_
    tuned_model.fit(X_train, y_train)
    y_pred_tuned = tuned_model.predict(X_test)

# Evaluate tuned model
acc_tuned = accuracy_score(y_test, y_pred_tuned)
f1_tuned = f1_score(y_test, y_pred_tuned, average='weighted', zero_division=0)

acc_before = results[best_model_name]['accuracy']
f1_before = results[best_model_name]['f1_score']

print(f"\n[5.2] Tuning Results ({best_model_name}):")
print(f"      Before tuning : Accuracy={acc_before*100:.2f}%, F1={f1_before:.4f}")
print(f"      After tuning  : Accuracy={acc_tuned*100:.2f}%, F1={f1_tuned:.4f}")
print(f"      Improvement   : Accuracy +{(acc_tuned-acc_before)*100:.2f}%, F1 +{(f1_tuned-f1_before):.4f}")

# ============================================================
# SECTION 6: CROSS-VALIDATION
# ============================================================
print("\n" + "=" * 70)
print("  SECTION 6: CROSS-VALIDATION")
print("=" * 70)

print(f"\n[6.1] Running 5-Fold Cross-Validation on tuned {best_model_name}...")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

from sklearn.base import clone

if best_model_name == 'XGBoost':
    le_cv = LabelEncoder()
    y_all_enc = le_cv.fit_transform(y)
    cv_model = clone(tuned_model)
    cv_scores = cross_val_score(cv_model, X, y_all_enc, cv=cv, scoring='accuracy')
else:
    cv_model = clone(tuned_model)
    cv_scores = cross_val_score(cv_model, X, y, cv=cv, scoring='accuracy')

print(f"\n      Fold scores : {[f'{s*100:.1f}%' for s in cv_scores]}")
print(f"      Mean accuracy: {cv_scores.mean()*100:.2f}% (+/- {cv_scores.std()*100:.2f}%)")
print(f"      95% CI       : [{(cv_scores.mean()-2*cv_scores.std())*100:.2f}%, {(cv_scores.mean()+2*cv_scores.std())*100:.2f}%]")

if cv_scores.std() < 0.03:
    print("      --> Model STABIL (std < 3%). Tidak overfit.")
elif cv_scores.std() < 0.06:
    print("      --> Model cukup stabil (std 3-6%).")
else:
    print("      --> WARNING: Model kurang stabil (std > 6%). Coba tambah data.")

# --- Learning Curve ---
print("\n[6.2] Generating learning curve...")

if best_model_name == 'XGBoost':
    lc_model = clone(tuned_model)
    train_sizes, train_scores, val_scores = learning_curve(
        lc_model, X, y_all_enc, cv=5, scoring='accuracy',
        train_sizes=np.linspace(0.1, 1.0, 10), n_jobs=-1
    )
else:
    lc_model = clone(tuned_model)
    train_sizes, train_scores, val_scores = learning_curve(
        lc_model, X, y, cv=5, scoring='accuracy',
        train_sizes=np.linspace(0.1, 1.0, 10), n_jobs=-1
    )

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(train_sizes, train_scores.mean(axis=1)*100, 'o-', label='Training Score', color='#3498db')
ax.plot(train_sizes, val_scores.mean(axis=1)*100, 'o-', label='Validation Score', color='#e74c3c')
ax.fill_between(train_sizes,
                val_scores.mean(axis=1)*100 - val_scores.std(axis=1)*100,
                val_scores.mean(axis=1)*100 + val_scores.std(axis=1)*100,
                alpha=0.1, color='#e74c3c')
ax.set_xlabel('Training Set Size')
ax.set_ylabel('Accuracy (%)')
ax.set_title(f'Learning Curve - {best_model_name}')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("data/learning_curve.png", dpi=150, bbox_inches='tight')
print("    Saved learning curve to: data/learning_curve.png")

# ============================================================
# SAVE BEST MODEL
# ============================================================
print("\n" + "=" * 70)
print("  SAVING BEST MODEL")
print("=" * 70)

os.makedirs("models", exist_ok=True)

# Save tuned model
joblib.dump(tuned_model, f"models/best_model_{best_model_name.replace(' ', '_').replace('(', '').replace(')', '')}.pkl")
joblib.dump(tfidf, "models/tfidf_vectorizer_improved.pkl")
joblib.dump(stemmer, "models/stemmer.pkl")
joblib.dump(stopword_remover, "models/stopword_remover.pkl")

# Save model metadata
metadata = {
    'model_name': best_model_name,
    'accuracy': acc_tuned,
    'f1_score': f1_tuned,
    'cv_mean': cv_scores.mean(),
    'cv_std': cv_scores.std(),
    'best_params': {k: str(v) for k, v in grid_search.best_params_.items()},
    'n_features': X.shape[1],
    'n_samples': len(df),
    'labels': list(set(y)),
}

with open("models/model_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print(f"\n  Model saved: models/best_model_{best_model_name.replace(' ', '_').replace('(', '').replace(')', '')}.pkl")
print(f"  Vectorizer saved: models/tfidf_vectorizer_improved.pkl")
print(f"  Stemmer saved: models/stemmer.pkl")
print(f"  Metadata saved: models/model_metadata.json")

# Save all comparison results
comparison_df = pd.DataFrame([
    {'Model': name, 'Accuracy': m['accuracy'], 'Precision': m['precision'],
     'Recall': m['recall'], 'F1_Score': m['f1_score']}
    for name, m in results.items()
])
comparison_df.to_csv("data/model_comparison.csv", index=False)
print(f"  Comparison saved: data/model_comparison.csv")

# Also save the best model as the default
joblib.dump(tuned_model, "models/best_model.pkl")
print(f"  Best model also saved as: models/best_model.pkl")

print("\n" + "=" * 70)
print(f"  TRAINING PIPELINE SELESAI!")
print(f"  Best Model: {best_model_name}")
print(f"  Accuracy  : {acc_tuned*100:.2f}%")
print(f"  F1-Score  : {f1_tuned:.4f}")
print(f"  CV Mean   : {cv_scores.mean()*100:.2f}% (+/- {cv_scores.std()*100:.2f}%)")
print("=" * 70)
