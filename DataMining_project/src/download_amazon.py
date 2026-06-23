"""
Amazon Electronics Reviews Downloader
=======================================
Downloads real Amazon product reviews from the McAuley-Lab/Amazon-Reviews-2023
dataset hosted on HuggingFace, using huggingface_hub for direct file streaming.

Streams large JSONL files and collects reviews for our 4 electronics categories:
phones, laptops, headphones, tablets.

Output: data/raw/amazon_reviews.csv

Usage:
    python src/download_amazon.py
"""

import pandas as pd
import json
import os
import sys
from datetime import datetime
from tqdm import tqdm

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "raw")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "amazon_reviews.csv")

# HuggingFace repo files to download
HF_REPO = "McAuley-Lab/Amazon-Reviews-2023"
HF_FILES = {
    "Cell_Phones_and_Accessories": "raw/review_categories/Cell_Phones_and_Accessories.jsonl",
    "Electronics": "raw/review_categories/Electronics.jsonl",
    "Computers": "raw/review_categories/Computers.jsonl",
}

# How many reviews to keep per source file (these files are HUGE — millions of lines)
MAX_PER_SOURCE = 3000
# Max lines to scan before giving up on a source
MAX_LINES_TO_SCAN = 100_000

# Keywords to classify reviews into our 4 project categories
PHONE_KW = ["phone", "smartphone", "iphone", "galaxy s", "galaxy a", "pixel", "oneplus",
            "xiaomi", "redmi", "motorola", "samsung galaxy", "cell phone"]
LAPTOP_KW = ["laptop", "notebook", "macbook", "chromebook", "thinkpad", "rog",
             "dell xps", "pavilion", "ideapad", "lenovo legion", "surface laptop"]
HEADPHONE_KW = ["headphone", "earphone", "earbud", "airpod", "earbuds", "headset",
                "in-ear", "over-ear", "beats", "bose", "sony wh", "sony wf", "jbl"]
TABLET_KW = ["tablet", "ipad", "galaxy tab", "surface pro", "kindle fire",
             "lenovo tab", "matepad", "xiaomi pad"]


def classify_review(title: str, text: str, source_category: str) -> str:
    """Classify a review into phones/laptops/headphones/tablets or None."""
    combined = (title + " " + text).lower()

    if source_category == "Cell_Phones_and_Accessories":
        if any(kw in combined for kw in TABLET_KW):
            return "tablets"
        if any(kw in combined for kw in HEADPHONE_KW):
            return "headphones"
        return "phones"

    if source_category == "Computers":
        if any(kw in combined for kw in TABLET_KW):
            return "tablets"
        if any(kw in combined for kw in LAPTOP_KW):
            return "laptops"
        return None  # skip desktops/peripherals

    if source_category == "Electronics":
        if any(kw in combined for kw in HEADPHONE_KW):
            return "headphones"
        if any(kw in combined for kw in TABLET_KW):
            return "tablets"
        if any(kw in combined for kw in PHONE_KW):
            return "phones"
        if any(kw in combined for kw in LAPTOP_KW):
            return "laptops"
        return None  # skip unrelated electronics

    return None


def download_and_parse(source_category: str, filepath: str, max_reviews: int):
    """Download a JSONL file from HuggingFace and extract matching reviews."""
    from huggingface_hub import hf_hub_download

    reviews = []
    lines_read = 0
    skipped = 0

    print(f"    Downloading {filepath.split('/')[-1]} from HuggingFace ...")

    try:
        # Download the file (cached locally after first download)
        local_path = hf_hub_download(
            repo_id=HF_REPO,
            filename=filepath,
            repo_type="dataset",
        )
        print(f"    File cached at: ...{local_path[-50:]}")

        # Stream parse the JSONL file
        file_size = os.path.getsize(local_path)
        print(f"    File size: {file_size / (1024*1024):.0f} MB — scanning up to {MAX_LINES_TO_SCAN:,} lines ...")

        with open(local_path, "r", encoding="utf-8") as f:
            for line in tqdm(f, desc=f"    Scanning", leave=False):
                if len(reviews) >= max_reviews:
                    break
                if lines_read >= MAX_LINES_TO_SCAN:
                    break

                lines_read += 1
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue

                text = item.get("text", "") or ""
                title = item.get("title", "") or ""
                rating = item.get("rating", 0)

                if len(text) < 30:
                    skipped += 1
                    continue

                category = classify_review(title, text, source_category)
                if category is None:
                    skipped += 1
                    continue

                timestamp = item.get("timestamp", 0)
                date_str = ""
                if timestamp:
                    try:
                        date_str = datetime.fromtimestamp(timestamp / 1000).strftime("%Y-%m-%d")
                    except (OSError, ValueError):
                        pass

                reviews.append({
                    "product_name": title[:100] if title else f"Amazon Product {item.get('asin', '')}",
                    "category": category,
                    "product_price": "",
                    "review_title": title[:200],
                    "review_text": text[:5000],
                    "star_rating": int(rating) if rating else 3,
                    "review_date": date_str,
                    "reviewer_name": (item.get("user_id", "") or "")[:20],
                    "source": "amazon",
                    "reddit_score": "",
                })

    except Exception as e:
        print(f"    ⚠ Error: {e}")

    print(f"    Lines scanned: {lines_read:,}, Skipped: {skipped:,}, Kept: {len(reviews):,}")
    return reviews


def download_amazon_reviews():
    """Download Amazon electronics reviews from HuggingFace."""
    print("=" * 60)
    print("  Amazon Electronics Reviews Downloader")
    print(f"  Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("  Source: McAuley-Lab/Amazon-Reviews-2023 (HuggingFace)")
    print("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    all_reviews = []

    for source_cat, filepath in HF_FILES.items():
        print(f"\n  [{source_cat}]")
        reviews = download_and_parse(source_cat, filepath, MAX_PER_SOURCE)
        all_reviews.extend(reviews)
        print(f"    ✅ Running total: {len(all_reviews):,} reviews")

    # Save results
    print(f"\n{'-' * 60}")
    print(f"  Total reviews collected: {len(all_reviews):,}")

    if all_reviews:
        df = pd.DataFrame(all_reviews)
        df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
        print(f"\n  ✅ Saved {len(df):,} reviews to {OUTPUT_FILE}")
        print(f"  Categories: {df['category'].value_counts().to_dict()}")
        print(f"  Rating distribution: {df['star_rating'].value_counts().sort_index().to_dict()}")
    else:
        print("  ⚠ No reviews collected.")

    print(f"\n  Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)


if __name__ == "__main__":
    download_amazon_reviews()
