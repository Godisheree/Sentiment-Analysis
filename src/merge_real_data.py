import pandas as pd
import numpy as np
import re

print("=" * 60)
print("  MERGE REAL + SYNTHETIC DATA")
print("=" * 60)

# ============================================================
# 1. LOAD REAL DATASETS
# ============================================================
print("\n[1] Loading real datasets...")

# Cellular (telco tweets)
df_cell = pd.read_csv("data/real_cellular.csv", encoding="latin-1")
df_cell = df_cell.rename(columns={"Text Tweet": "text", "Sentiment": "sentiment"})
df_cell = df_cell[["text", "sentiment"]]
print(f"    Cellular : {len(df_cell)} tweets")

# Pilkada (election tweets)
df_pil = pd.read_csv("data/real_pilkada.csv", encoding="latin-1")
df_pil = df_pil.rename(columns={"Text Tweet": "text", "Sentiment": "sentiment"})
df_pil = df_pil[["text", "sentiment"]]
print(f"    Pilkada  : {len(df_pil)} tweets")

# TV shows
df_tv = pd.read_csv("data/real_tv.csv", encoding="latin-1")
df_tv = df_tv.rename(columns={"Text Tweet": "text", "Sentiment": "sentiment"})
df_tv = df_tv[["text", "sentiment"]]
print(f"    TV Shows : {len(df_tv)} tweets")

# Combine real datasets
df_real = pd.concat([df_cell, df_pil, df_tv], ignore_index=True)
df_real = df_real.drop_duplicates(subset=["text"])
print(f"    Real total (dedup): {len(df_real)} tweets")

# ============================================================
# 2. CLEAN REAL TWEETS
# ============================================================
print("\n[2] Cleaning real tweets...")

def clean_tweet(text):
    """Clean tweet: remove placeholders, normalize."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r'<USER_MENTION>', '', text)
    text = re.sub(r'<PROVIDER_NAME>', '', text)
    text = re.sub(r'<URL>', '', text)
    text = re.sub(r'<NUMBER>', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'http\S+|www\.\S+', '', text)
    text = re.sub(r'RT\s+', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

df_real["text"] = df_real["text"].apply(clean_tweet)
df_real = df_real[df_real["text"].str.len() > 10].reset_index(drop=True)
print(f"    After cleaning (min 10 chars): {len(df_real)} tweets")

# ============================================================
# 3. ADD NEUTRAL SAMPLES FROM REAL DATA
# ============================================================
print("\n[3] Generating neutral samples from real data...")

# Strategy: tweets that are short, factual, question-like -> label as neutral
neutral_keywords = ["ada", "bisa", "mohon", "tolong", "mau tanya", "info",
                    "berapa", "kapan", "dimana", "bagaimana", "kenapa",
                    "apakah", "mohon info", "bantu"]

neutral_candidates = []
for idx, row in df_real.iterrows():
    text_lower = row["text"].lower()
    if any(kw in text_lower for kw in neutral_keywords):
        neutral_candidates.append({"text": row["text"], "sentiment": "neutral"})

df_neutral_real = pd.DataFrame(neutral_candidates)
print(f"    Neutral samples extracted: {len(df_neutral_real)}")

# Remove these from positive/negative to avoid duplicates
if len(df_neutral_real) > 0:
    neutral_texts = set(df_neutral_real["text"])
    df_real = df_real[~df_real["text"].isin(neutral_texts)]

# ============================================================
# 4. LOAD EXISTING SYNTHETIC DATA
# ============================================================
print("\n[4] Loading existing synthetic data...")
df_synth = pd.read_csv("data/combined_sentiment.csv")
print(f"    Synthetic: {len(df_synth)} samples")

# ============================================================
# 5. COMBINE ALL
# ============================================================
print("\n[5] Combining all datasets...")

df_all = pd.concat([df_synth, df_real, df_neutral_real], ignore_index=True)
df_all = df_all.drop_duplicates(subset=["text"])

print(f"\n    FINAL DATASET:")
print(f"    Total samples: {len(df_all)}")
print(f"\n    Class distribution:")
for sent, count in df_all["sentiment"].value_counts().items():
    pct = count / len(df_all) * 100
    print(f"      {sent:12s} : {count:4d} ({pct:.1f}%)")

# Source breakdown
print(f"\n    Sources:")
print(f"      Synthetic (template) : {len(df_synth)}")
print(f"      Real tweets          : {len(df_real)}")
print(f"      Real neutral         : {len(df_neutral_real)}")

# Text stats
df_all["text_len"] = df_all["text"].str.len()
print(f"\n    Text length stats:")
print(f"      Mean: {df_all['text_len'].mean():.1f} chars")
print(f"      Min : {df_all['text_len'].min()} chars")
print(f"      Max : {df_all['text_len'].max()} chars")

# Sample real tweets
print(f"\n    Sample REAL tweets:")
for sent in ["positive", "negative"]:
    sample = df_real[df_real["sentiment"] == sent].iloc[0]["text"][:80]
    print(f"      [{sent:8s}] {sample}...")
if len(df_neutral_real) > 0:
    sample = df_neutral_real.iloc[0]["text"][:80]
    print(f"      [neutral ] {sample}...")

# ============================================================
# 6. SAVE
# ============================================================
print("\n[6] Saving combined dataset...")
df_all[["text", "sentiment"]].to_csv("data/real_combined_sentiment.csv", index=False)
print(f"    Saved to: data/real_combined_sentiment.csv ({len(df_all)} samples)")

print(f"\n{'=' * 60}")
print(f"  DONE! {len(df_all)} total samples (synthetic + real)")
print(f"{'=' * 60}")
