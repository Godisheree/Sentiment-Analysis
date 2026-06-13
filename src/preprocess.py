import pandas as pd
import re

print("=" * 60)
print("  DATA CLEANING & PREPROCESSING")
print("  Sentiment Analysis - Indonesian Dataset")
print("=" * 60)

# --- 1. Load Raw Data ---
print("\n[1] Memuat data mentah...")
df = pd.read_csv("data/raw_sentiment.csv")
print(f"    Data mentah dimuat: {len(df)} baris")

# --- 2. Bersihkan Teks ---
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
    text = text.strip()

    return text

print("\n[2] Membersihkan teks...")
df['clean_text'] = df['text'].apply(clean_text)

# --- 3. Tampilkan Hasil Cleaning ---
print("\n[3] Contoh Hasil Cleaning:")
for i in [0, 50, 100]:
    if i < len(df):
        print(f"\n    Sebelum : {df['text'].iloc[i]}")
        print(f"    Sesudah : {df['clean_text'].iloc[i]}")

# --- 4. Hapus Baris Kosong ---
before = len(df)
df = df[df['clean_text'].str.len() > 0]
removed = before - len(df)
print(f"\n[4] Baris dihapus (teks kosong): {removed}")
print(f"    Data tersisa: {len(df)} baris")

# --- 5. Statistik Teks Bersih ---
df['clean_length'] = df['clean_text'].str.split().apply(len)
print("\n[5] Statistik Jumlah Kata (setelah cleaning):")
print(f"    Rata-rata  : {df['clean_length'].mean():.1f} kata")
print(f"    Minimum    : {df['clean_length'].min()} kata")
print(f"    Maksimum   : {df['clean_length'].max()} kata")

# --- 6. Simpan Data Bersih ---
df_output = df[['text', 'clean_text', 'sentiment']].copy()
output_path = "data/cleaned_data.csv"
df_output.to_csv(output_path, index=False)
print(f"\n[6] Data bersih disimpan ke: {output_path}")
print(f"    Total data bersih: {len(df_output)} baris")

# --- 7. Distribusi Setelah Cleaning ---
print("\n[7] Distribusi Sentimen (setelah cleaning):")
for sentiment, count in df['sentiment'].value_counts().items():
    print(f"    {sentiment:12s} : {count} data")

print("\n" + "=" * 60)
print("  PREPROCESSING SELESAI!")
print("  File output: data/cleaned_data.csv")
print("=" * 60)
