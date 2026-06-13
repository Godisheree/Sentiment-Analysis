import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

print("=" * 60)
print("  EXPLORATORY DATA ANALYSIS (EDA)")
print("  Sentiment Analysis - Indonesian Dataset")
print("=" * 60)

# --- 1. Load Dataset ---
print("\n[1] Memuat dataset...")
df = pd.read_csv("data/raw_sentiment.csv")
print(f"    Dataset berhasil dimuat: {len(df)} baris, {len(df.columns)} kolom")

# --- 2. Info Dasar ---
print("\n[2] Informasi Dataset:")
print(f"    Jumlah data     : {len(df)}")
print(f"    Kolom-kolom     : {list(df.columns)}")
print(f"    Tipe data       :\n{df.dtypes.to_string()}")

# --- 3. Sample Data ---
print("\n[3] 5 Sample Data Pertama:")
print(df.head().to_string())

# --- 4. Distribusi Sentimen ---
print("\n[4] Distribusi Sentimen:")
sentiment_counts = df['sentiment'].value_counts()
for sentiment, count in sentiment_counts.items():
    percentage = (count / len(df)) * 100
    print(f"    {sentiment:12s} : {count:3d} data ({percentage:.1f}%)")

# --- 5. Missing Values ---
print("\n[5] Missing Values:")
missing = df.isnull().sum()
for col in df.columns:
    print(f"    {col:12s} : {missing[col]} missing")

total_missing = missing.sum()
if total_missing == 0:
    print("    --> Tidak ada missing values. Dataset bersih!")
else:
    print(f"    --> Total {total_missing} missing values ditemukan.")

# --- 6. Statistik Panjang Teks ---
df['text_length'] = df['text'].str.len()
print("\n[6] Statistik Panjang Teks (karakter):")
print(f"    Rata-rata  : {df['text_length'].mean():.1f}")
print(f"    Minimum    : {df['text_length'].min()}")
print(f"    Maksimum   : {df['text_length'].max()}")

# --- 7. Visualisasi ---
print("\n[7] Membuat visualisasi...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Bar chart distribusi sentimen
colors = {'positive': '#2ecc71', 'negative': '#e74c3c', 'neutral': '#95a5a6'}
bar_colors = [colors.get(s, '#333333') for s in sentiment_counts.index]
axes[0].bar(sentiment_counts.index, sentiment_counts.values, color=bar_colors, edgecolor='black')
axes[0].set_title('Distribusi Sentimen', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Sentimen')
axes[0].set_ylabel('Jumlah Data')
for i, v in enumerate(sentiment_counts.values):
    axes[0].text(i, v + 1, str(v), ha='center', fontweight='bold')

# Pie chart
axes[1].pie(sentiment_counts.values, labels=sentiment_counts.index,
            autopct='%1.1f%%', colors=bar_colors, startangle=90)
axes[1].set_title('Proporsi Sentimen', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig("data/eda_visualization.png", dpi=150, bbox_inches='tight')
print("    Visualisasi disimpan ke: data/eda_visualization.png")

# --- 8. Sample per Sentimen ---
print("\n[8] Contoh Teks per Sentimen:")
for sentiment in ['positive', 'negative', 'neutral']:
    sample = df[df['sentiment'] == sentiment].iloc[0]['text']
    print(f"    [{sentiment.upper():8s}] {sample}")

print("\n" + "=" * 60)
print("  EDA SELESAI!")
print("=" * 60)
