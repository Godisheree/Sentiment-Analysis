"""Scrape real Indonesian reviews from Google Play Store.

Collects reviews from popular Indonesian apps and labels sentiment
based on star rating: 5 stars = positive, 1-2 stars = negative,
3-4 stars = neutral.
"""

import pandas as pd
import re
from google_play_scraper import Sort, reviews_all

APPS = {
    "Gojek": "com.gojek.app",
    "Tokopedia": "com.tokopedia.android",
    "Shopee Indonesia": "com.shopee.id",
}

REVIEWS_PER_APP = 100
MIN_TEXT_LENGTH = 15
MAX_TEXT_LENGTH = 500


def clean_review(text: str) -> str:
    """Clean a review text."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r"\s+", " ", text).strip()
    return text


def is_indonesian(text: str) -> bool:
    """Basic heuristic: contains common Indonesian words."""
    indonesian_markers = [
        "yang", "dan", "di", "ke", "dari", "untuk", "dengan", "ini", "itu",
        "saya", "sudah", "belum", "bisa", "tidak", "ada", "juga", "sangat",
        "bagus", "baik", "buruk", "jelek", "aplikasi", "bagus", "keren",
        "mudah", "susah", "cepat", "lambat", "puas", "kecewa",
    ]
    text_lower = text.lower()
    return any(word in text_lower for word in indonesian_markers)


def star_to_sentiment(star: int) -> str:
    """Convert star rating to sentiment label."""
    if star == 5:
        return "positive"
    elif star <= 2:
        return "negative"
    else:
        return "neutral"


def scrape_app(app_name: str, app_id: str) -> pd.DataFrame:
    """Scrape reviews from a single app."""
    print(f"  Scraping {app_name} ({app_id})...")

    try:
        result = reviews_all(
            app_id,
            lang="id",
            country="id",
            sort=Sort.NEWEST,
            count=REVIEWS_PER_APP,
        )
    except Exception as e:
        print(f"    ERROR scraping {app_name}: {e}")
        return pd.DataFrame()

    if not result:
        print(f"    No reviews found for {app_name}")
        return pd.DataFrame()

    rows = []
    for review in result:
        text = clean_review(review.get("content", ""))
        if len(text) < MIN_TEXT_LENGTH or len(text) > MAX_TEXT_LENGTH:
            continue
        if not is_indonesian(text):
            continue

        star = review.get("score", 3)
        sentiment = star_to_sentiment(star)

        rows.append({
            "text": text,
            "sentiment": sentiment,
            "star": star,
            "source": app_name,
        })

    df = pd.DataFrame(rows)
    print(f"    Got {len(df)} valid Indonesian reviews")
    return df


def main():
    print("=" * 70)
    print("  SCRAPING REAL INDONESIAN REVIEWS FROM GOOGLE PLAY")
    print("=" * 70)

    all_reviews = []

    for app_name, app_id in APPS.items():
        df = scrape_app(app_name, app_id)
        if len(df) > 0:
            all_reviews.append(df)

    if not all_reviews:
        print("\n  ERROR: No reviews collected from any app!")
        return

    combined = pd.concat(all_reviews, ignore_index=True)
    combined = combined.drop_duplicates(subset=["text"])

    print(f"\n{'=' * 70}")
    print(f"  SCRAPING RESULTS")
    print(f"{'=' * 70}")
    print(f"  Total reviews: {len(combined)}")
    print(f"\n  Sentiment distribution:")
    for sent, count in combined["sentiment"].value_counts().items():
        pct = count / len(combined) * 100
        print(f"    {sent:12s} : {count:4d} ({pct:.1f}%)")
    print(f"\n  Source breakdown:")
    for source, count in combined["source"].value_counts().items():
        print(f"    {source:20s} : {count:4d}")

    combined[["text", "sentiment"]].to_csv("data/google_play_reviews.csv", index=False)
    print(f"\n  Saved to: data/google_play_reviews.csv ({len(combined)} reviews)")

    print(f"\n{'=' * 70}")
    print(f"  SCRAPING COMPLETE!")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
