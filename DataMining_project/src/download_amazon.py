"""
Amazon Electronics Reviews Downloader
=======================================
Extracts real Amazon product reviews from the McAuley-Lab/Amazon-Reviews-2023
dataset. Uses the locally cached HuggingFace file (already downloaded).

Reads the Cell_Phones_and_Accessories JSONL and classifies reviews into
our 4 electronics categories: phones, laptops, headphones, tablets.

Output: data/raw/amazon_reviews.csv

Usage:
    python src/download_amazon.py
"""

import pandas as pd
import json
import os
import sys
from datetime import datetime

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "raw")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "amazon_reviews.csv")

# HuggingFace repo info (we use hf_hub_download to get/check the cached file)
HF_REPO = "McAuley-Lab/Amazon-Reviews-2023"
HF_FILE = "raw/review_categories/Cell_Phones_and_Accessories.jsonl"

# Scan more lines to get enough diverse reviews from this one big file
MAX_REVIEWS = 9000
MAX_LINES_TO_SCAN = 500_000

# Keywords to classify reviews into our 4 project categories
PHONE_KW = ["phone", "smartphone", "iphone", "galaxy s", "galaxy a", "pixel", "oneplus",
            "xiaomi", "redmi", "motorola", "samsung galaxy", "cell phone", "mobile"]
LAPTOP_KW = ["laptop", "notebook", "macbook", "chromebook", "thinkpad",
             "dell xps", "pavilion", "ideapad", "surface laptop"]
HEADPHONE_KW = ["headphone", "earphone", "earbud", "airpod", "earbuds", "headset",
                "in-ear", "over-ear", "beats", "bose", "sony wh", "sony wf", "jbl",
                "wireless earbuds", "bluetooth earbuds", "noise cancelling"]
TABLET_KW = ["tablet", "ipad", "galaxy tab", "surface pro", "kindle fire",
             "lenovo tab", "fire hd", "fire tablet"]


def classify_review(title, text):
    """Classify a review into phones/laptops/headphones/tablets."""
    combined = (title + " " + text).lower()

    # Check specific categories first (more specific keywords)
    if any(kw in combined for kw in HEADPHONE_KW):
        return "headphones"
    if any(kw in combined for kw in TABLET_KW):
        return "tablets"
    if any(kw in combined for kw in LAPTOP_KW):
        return "laptops"
    # Default for Cell_Phones category: phones
    return "phones"


def download_amazon_reviews():
    """Extract Amazon electronics reviews from cached HuggingFace file."""
    from huggingface_hub import hf_hub_download

    print("=" * 60)
    print("  Amazon Electronics Reviews Downloader")
    print("  Started at: {}".format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
    print("  Source: McAuley-Lab/Amazon-Reviews-2023 (HuggingFace)")
    print("=" * 60)
    sys.stdout.flush()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Download / use cached file
    print("\n  Downloading Cell_Phones_and_Accessories.jsonl ...")
    print("  (This file is ~8.9 GB, first download takes time, then cached)")
    sys.stdout.flush()

    local_path = hf_hub_download(
        repo_id=HF_REPO,
        filename=HF_FILE,
        repo_type="dataset",
    )

    file_size = os.path.getsize(local_path)
    print("  File cached: {:.0f} MB".format(file_size / (1024*1024)))
    print("  Scanning up to {:,} lines for {:,} reviews ...".format(MAX_LINES_TO_SCAN, MAX_REVIEWS))
    sys.stdout.flush()

    reviews = []
    lines_read = 0
    skipped = 0

    with open(local_path, "r", encoding="utf-8", buffering=8*1024*1024) as f:
        for line in f:
            if len(reviews) >= MAX_REVIEWS:
                break
            if lines_read >= MAX_LINES_TO_SCAN:
                break

            lines_read += 1

            if lines_read % 25000 == 0:
                print("    ... scanned {:,} lines, kept {:,} reviews".format(lines_read, len(reviews)))
                sys.stdout.flush()

            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue

            text = item.get("text", "") or ""
            title = item.get("title", "") or ""
            rating = item.get("rating", 0)

            # Skip very short reviews
            if len(text) < 30:
                skipped += 1
                continue

            category = classify_review(title, text)

            timestamp = item.get("timestamp", 0) or item.get("sort_timestamp", 0)
            date_str = ""
            if timestamp:
                try:
                    date_str = datetime.fromtimestamp(timestamp / 1000).strftime("%Y-%m-%d")
                except (OSError, ValueError):
                    pass

            reviews.append({
                "product_name": f"Product {item.get('asin', '')}",
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

    # Save results
    print("\n" + "-" * 60)
    print("  Lines scanned: {:,}".format(lines_read))
    print("  Skipped (too short): {:,}".format(skipped))
    print("  Total reviews collected: {:,}".format(len(reviews)))

    if reviews:
        df = pd.DataFrame(reviews)
        df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
        print("\n  Saved {:,} reviews to {}".format(len(df), OUTPUT_FILE))
        print("  Categories: {}".format(df['category'].value_counts().to_dict()))
        print("  Rating distribution: {}".format(df['star_rating'].value_counts().sort_index().to_dict()))
    else:
        print("  No reviews collected.")

    print("\n  Finished at: {}".format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
    print("=" * 60)
    sys.stdout.flush()


if __name__ == "__main__":
    download_amazon_reviews()
