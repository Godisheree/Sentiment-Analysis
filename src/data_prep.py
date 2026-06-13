import pandas as pd
import numpy as np
import random
import os

random.seed(42)
np.random.seed(42)

print("=" * 60)
print("  SECTION 1: GATHER MORE TRAINING DATA")
print("  Generating Enhanced Indonesian Sentiment Dataset")
print("=" * 60)

# ============================================================
# 1. TEMPLATE-BASED DATA GENERATION
# ============================================================
print("\n[1] Generating synthetic Indonesian sentiment data...")

positive_adjectives = [
    "bagus", "hebat", "luar biasa", "mantap", "keren", "top", "prima",
    "terbaik", "berkualitas", "puas", "senang", "nyaman", "mudah",
    "cepat", "efisien", "praktis", "murah", "terjangkau", "awet",
    "tahan lama", "elegan", "cantik", "rapi", "bersih", "wangi",
    "lezat", "enak", "segar", "halus", "lembut", "kuat", "kokoh",
    "stabil", "smooth", "recommended", "worth it", "sangat puas",
    "luar biasa bagus", "sangat membantu", "sangat recommended",
    "benar-benar puas", "super puas", "mantap jiwa", "jos", "oke banget",
    "parah bagusnya", "gila keren", "edun bagusnya"
]

negative_adjectives = [
    "buruk", "jelek", "rusak", "cacat", "kecewa", "menyesal", "mahal",
    "lambat", "lemot", "payah", "teruk", "sampah", "bohong", "palsu",
    "tipu", "kotor", "bau", "amis", "kasar", "tipis", "rapuh",
    "mudah rusak", "tidak berguna", "tidak berfungsi", "mengecewakan",
    "tidak puas", "sangat kecewa", "sangat buruk", "tidak worth it",
    "benar-benar kecewa", "nyesel beli", "buang uang", "rugi",
    "parah", "gila jeleknya", "kacau", "bobrok", "ngaco", "amburadul",
    "zonk", "tidak jelas", "abal-abal", "kw", "palsu semua"
]

neutral_phrases = [
    "biasa saja", "lumayan", "cukup oke", "standar", "normal",
    "tidak ada yang spesial", "sesuai harga", "cukup baik",
    "tidak mengecewakan", "tidak istimewa", "rata-rata", "so-so",
    "cukup", "mayoritas oke", "tidak ada masalah", "b aja",
    "biasa aja", "tidak terlalu wow", "cukup puas", "lumayan lah"
]

positive_nouns = ["produk", "barang", "pelayanan", "pengiriman", "packaging", "kualitas", "bahan", "desain", "fungsi", "seller"]
negative_nouns = ["produk", "barang", "pelayanan", "pengiriman", "packaging", "kualitas", "bahan", "seller", "toko", "kurir"]
neutral_nouns = ["produk", "barang", "pelayanan", "pengiriman", "kualitas", "harga", "seller", "toko"]

# --- Template functions ---
def gen_positive():
    templates = [
        lambda: f"{random.choice(positive_nouns)} ini {random.choice(positive_adjectives)}",
        lambda: f"{random.choice(positive_nouns)} {random.choice(positive_adjectives)} sekali",
        lambda: f"sangat {random.choice(positive_adjectives)} {random.choice(positive_nouns)}nya",
        lambda: f"saya {random.choice(['puas', 'senang', 'terpuaskan'])} dengan {random.choice(positive_nouns)} ini",
        lambda: f"{random.choice(positive_nouns)}nya {random.choice(positive_adjectives)} dan {random.choice(positive_adjectives)}",
        lambda: f"tidak menyesal beli {random.choice(positive_nouns)} ini {random.choice(positive_adjectives)} banget",
        lambda: f"{random.choice(positive_adjectives)} sekali {random.choice(positive_nouns)} ini recommended",
        lambda: f"pengalaman {random.choice(['belanja', 'memakai'])} yang {random.choice(positive_adjectives)}",
        lambda: f"{random.choice(positive_nouns)} sangat {random.choice(positive_adjectives)} pasti beli lagi",
        lambda: f"kualitas {random.choice(positive_adjectives)} harga {random.choice(['terjangkau', 'murah', 'bersahabat'])}",
        lambda: f"barang {random.choice(positive_adjectives)} pengiriman {random.choice(['cepat', 'super cepat', 'lancar'])}",
        lambda: f"seller {random.choice(['ramah', 'responsif', 'amanah', 'baik'])} {random.choice(positive_nouns)} {random.choice(positive_adjectives)}",
        lambda: f"packaging {random.choice(['rapi', 'aman', 'bagus'])} {random.choice(positive_nouns)} {random.choice(positive_adjectives)}",
        lambda: f"ini {random.choice(positive_nouns)} terbaik yang pernah saya {random.choice(['beli', 'punya', 'coba'])}",
        lambda: f"recommended banget {random.choice(positive_nouns)}nya {random.choice(positive_adjectives)}",
    ]
    return random.choice(templates)()

def gen_negative():
    templates = [
        lambda: f"{random.choice(negative_nouns)} ini {random.choice(negative_adjectives)}",
        lambda: f"{random.choice(negative_nouns)} {random.choice(negative_adjectives)} sekali",
        lambda: f"sangat {random.choice(negative_adjectives)} dengan {random.choice(negative_nouns)} ini",
        lambda: f"saya {random.choice(['kecewa', 'menyesal', 'kesal'])} dengan {random.choice(negative_nouns)} ini",
        lambda: f"{random.choice(negative_nouns)}nya {random.choice(negative_adjectives)} dan {random.choice(negative_adjectives)}",
        lambda: f"rugi beli {random.choice(negative_nouns)} ini {random.choice(negative_adjectives)} banget",
        lambda: f"jangan beli di sini {random.choice(negative_nouns)}nya {random.choice(negative_adjectives)}",
        lambda: f"pengalaman {random.choice(['belanja', 'memakai'])} yang {random.choice(negative_adjectives)}",
        lambda: f"{random.choice(negative_nouns)} sangat {random.choice(negative_adjectives)} tidak recommended",
        lambda: f"barang {random.choice(negative_adjectives)} pengiriman {random.choice(['lama', 'sangat lama', 'telat'])}",
        lambda: f"seller {random.choice(['kabur', 'tidak responsif', 'cuek', 'bohong'])} {random.choice(negative_nouns)} {random.choice(negative_adjectives)}",
        lambda: f"penipuan {random.choice(negative_nouns)} tidak sesuai {random.choice(['gambar', 'deskripsi', 'foto'])}",
        lambda: f"{random.choice(negative_nouns)} datang dalam kondisi {random.choice(negative_adjectives)}",
        lambda: f"ini {random.choice(negative_nouns)} terburuk yang pernah saya {random.choice(['beli', 'punya', 'coba'])}",
        lambda: f"uang saya hilang {random.choice(negative_nouns)} {random.choice(negative_adjectives)} tidak ada solusi",
    ]
    return random.choice(templates)()

def gen_neutral():
    templates = [
        lambda: f"{random.choice(neutral_nouns)} ini {random.choice(neutral_phrases)}",
        lambda: f"{random.choice(neutral_nouns)} {random.choice(neutral_phrases)} untuk harganya",
        lambda: f"cukup {random.choice(['oke', 'baik', 'puas'])} dengan {random.choice(neutral_nouns)} ini",
        lambda: f"{random.choice(neutral_nouns)}nya {random.choice(neutral_phrases)} tidak ada yang spesial",
        lambda: f"{random.choice(neutral_nouns)} sudah sampai {random.choice(neutral_phrases)}",
        lambda: f"pengiriman {random.choice(['normal', 'standar', 'biasa'])} {random.choice(neutral_nouns)} {random.choice(neutral_phrases)}",
        lambda: f"tidak ada masalah dengan {random.choice(neutral_nouns)} ini {random.choice(neutral_phrases)}",
        lambda: f"{random.choice(neutral_nouns)} berfungsi {random.choice(['normal', 'baik', 'biasa'])}",
        lambda: f"kualitas {random.choice(['sesuai harga', 'standar', 'rata-rata'])} {random.choice(neutral_phrases)}",
        lambda: f"seller {random.choice(['biasa saja', 'cukup ramah', 'standar'])} {random.choice(neutral_nouns)} {random.choice(neutral_phrases)}",
        lambda: f"{random.choice(neutral_phrases)} saja {random.choice(neutral_nouns)} ini untuk pemula",
        lambda: f"beli {random.choice(neutral_nouns)} ini {random.choice(neutral_phrases)} tidak kecewa tidak juga wow",
    ]
    return random.choice(templates)()

# Generate data
data_list = []
n_per_class = 700

print(f"    Generating {n_per_class} samples per class...")

for _ in range(n_per_class):
    data_list.append({"text": gen_positive(), "sentiment": "positive"})
for _ in range(n_per_class):
    data_list.append({"text": gen_negative(), "sentiment": "negative"})
for _ in range(n_per_class):
    data_list.append({"text": gen_neutral(), "sentiment": "neutral"})

df_synthetic = pd.DataFrame(data_list)
# Remove exact duplicates
df_synthetic = df_synthetic.drop_duplicates(subset=['text'])
print(f"    Synthetic data generated: {len(df_synthetic)} unique samples")

# ============================================================
# 2. LOAD EXISTING DATASET
# ============================================================
print("\n[2] Loading existing dataset...")
df_existing = pd.read_csv("data/raw_sentiment.csv")
print(f"    Existing data: {len(df_existing)} samples")

# ============================================================
# 3. COMBINE DATASETS
# ============================================================
print("\n[3] Combining datasets...")
df_combined = pd.concat([df_existing, df_synthetic], ignore_index=True)
df_combined = df_combined.drop_duplicates(subset=['text'])
print(f"    Combined data (after dedup): {len(df_combined)} samples")

# ============================================================
# 4. VALIDATE COMBINED DATASET
# ============================================================
print("\n[4] Validating combined dataset...")
print(f"    Total samples : {len(df_combined)}")
print(f"    Columns       : {list(df_combined.columns)}")
print(f"    Missing values: {df_combined.isnull().sum().sum()}")

print("\n    Class distribution:")
for sentiment, count in df_combined['sentiment'].value_counts().items():
    pct = (count / len(df_combined)) * 100
    print(f"      {sentiment:12s} : {count:4d} ({pct:.1f}%)")

# Text length stats
df_combined['text_len'] = df_combined['text'].str.len()
print(f"\n    Text length stats:")
print(f"      Mean  : {df_combined['text_len'].mean():.1f} chars")
print(f"      Min   : {df_combined['text_len'].min()} chars")
print(f"      Max   : {df_combined['text_len'].max()} chars")

# ============================================================
# 5. SAVE COMBINED DATASET
# ============================================================
print("\n[5] Saving combined dataset...")
os.makedirs("data", exist_ok=True)
df_combined[['text', 'sentiment']].to_csv("data/combined_sentiment.csv", index=False)
print(f"    Saved to: data/combined_sentiment.csv ({len(df_combined)} samples)")

print("\n" + "=" * 60)
print(f"  SECTION 1 SELESAI!")
print(f"  Total data: {len(df_combined)} samples (target: 1000+)")
print("=" * 60)
