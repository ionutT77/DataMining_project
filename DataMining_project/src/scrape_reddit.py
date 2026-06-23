"""
Reddit Electronics Reviews Scraper (PRAW)
==========================================
Scrapes product review posts from electronics subreddits using the
official Reddit API via PRAW (Python Reddit API Wrapper).

SETUP (one-time, takes 1 minute):
  1. Go to https://www.reddit.com/prefs/apps
  2. Click "create another app..." at the bottom
  3. Name: "DataMiningProject" (anything)
  4. Type: select "script"
  5. Redirect URI: http://localhost:8080
  6. Click "create app"
  7. Copy the client_id (under the app name) and client_secret
  8. Set them below or as environment variables

Output: data/raw/reddit_reviews.csv

Usage:
    python src/scrape_reddit.py
"""

import praw
import pandas as pd
import os
import re
import sys
from datetime import datetime, timezone
from tqdm import tqdm

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

# ---------------------------------------------------------------------------
# ⚙ Reddit API Credentials — FILL THESE IN
# ---------------------------------------------------------------------------
# Option 1: Set directly here
REDDIT_CLIENT_ID = os.environ.get("REDDIT_CLIENT_ID", "YOUR_CLIENT_ID")
REDDIT_CLIENT_SECRET = os.environ.get("REDDIT_CLIENT_SECRET", "YOUR_CLIENT_SECRET")
REDDIT_USER_AGENT = "DataMiningProject/1.0 (Electronics Reviews Scraper)"

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SUBREDDITS = {
    "phones": ["smartphones", "Android", "iphone", "GooglePixel", "samsung", "oneplus"],
    "laptops": ["laptops", "SuggestALaptop", "thinkpad", "razer"],
    "headphones": ["headphones", "HeadphoneAdvice", "airpods"],
    "tablets": ["tablets", "ipad", "GalaxyTab"],
}

SEARCH_QUERIES = ["review", "my experience", "months later", "honest opinion", "worth it"]
MIN_BODY_LENGTH = 120
MAX_POSTS_PER_SUBREDDIT = 200

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "raw")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "reddit_reviews.csv")


# ---------------------------------------------------------------------------
# Sentiment-based star rating derivation
# ---------------------------------------------------------------------------
STRONG_POS = re.compile(
    r"\b(love|amazing|incredible|perfect|excellent|outstanding|fantastic|"
    r"phenomenal|blown away|game.?changer|best .{0,20} ever|highly recommend|"
    r"10/10|9/10|5/5)\b", re.I
)
POS = re.compile(
    r"\b(good|great|nice|solid|recommend|happy|satisfied|impressed|"
    r"enjoy|comfortable|worth|reliable|quality|8/10|4/5)\b", re.I
)
NEG = re.compile(
    r"\b(disappointed|disappointing|issues?|problems?|mediocre|underwhelm|"
    r"not great|could be better|overrated|overpriced|meh|3/10|4/10|2/5)\b", re.I
)
STRONG_NEG = re.compile(
    r"\b(terrible|worst|awful|hate|garbage|trash|broken|waste|junk|"
    r"don.?t buy|avoid|regret|return|refund|1/10|2/10|0/10|1/5)\b", re.I
)


def derive_star_rating(text: str) -> int:
    """Derive a 1-5 star rating from review text using keyword sentiment."""
    sp = len(STRONG_POS.findall(text))
    p = len(POS.findall(text))
    n = len(NEG.findall(text))
    sn = len(STRONG_NEG.findall(text))

    score = (sp * 2) + (p * 1) + (n * -1) + (sn * -2)

    # Check for explicit ratings like "8/10", "4/5"
    explicit = re.findall(r"(\d+)\s*/\s*(10|5)\b", text)
    for num, denom in explicit:
        ratio = int(num) / int(denom)
        if ratio >= 0.8:
            score += 2
        elif ratio >= 0.6:
            score += 1
        elif ratio <= 0.3:
            score -= 2
        elif ratio <= 0.5:
            score -= 1

    if score >= 3:
        return 5
    elif score >= 1:
        return 4
    elif score == 0:
        return 3
    elif score >= -2:
        return 2
    else:
        return 1


def extract_product_name(title: str) -> str:
    """Try to extract a product name from the post title."""
    cleaned = re.sub(r"^\[.*?\]\s*", "", title)
    cleaned = re.sub(
        r"^(my\s+)?(review|experience|thoughts|opinion|honest\s+review)"
        r"(\s+of|\s+on|\s+with|\s*:|\s*-)\s*",
        "", cleaned, flags=re.I
    )
    cleaned = cleaned.split(".")[0].split("!")[0].split("?")[0]
    return cleaned[:100].strip() or title[:80]


# ---------------------------------------------------------------------------
# Main scraper
# ---------------------------------------------------------------------------
def scrape_reddit_reviews():
    """Scrape electronics review posts from Reddit using PRAW."""
    print("=" * 60)
    print("  Reddit Electronics Reviews Scraper (PRAW)")
    print(f"  Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # Check credentials
    if REDDIT_CLIENT_ID == "YOUR_CLIENT_ID":
        print("\n  ⚠ Reddit API credentials not configured!")
        print("  To set up (takes 1 minute):")
        print("    1. Go to https://www.reddit.com/prefs/apps")
        print("    2. Click 'create another app...' at the bottom")
        print("    3. Name: 'DataMiningProject', Type: 'script'")
        print("    4. Redirect URI: http://localhost:8080")
        print("    5. Click 'create app'")
        print("    6. Copy client_id and client_secret")
        print("    7. Either edit REDDIT_CLIENT_ID/SECRET in this file")
        print("       or set env vars REDDIT_CLIENT_ID / REDDIT_CLIENT_SECRET")
        print("\n  Reddit's public JSON API blocks unauthenticated requests (403).")
        print("  PRAW with OAuth is the only reliable way to access Reddit data.\n")
        return 0

    # Initialize PRAW
    try:
        reddit = praw.Reddit(
            client_id=REDDIT_CLIENT_ID,
            client_secret=REDDIT_CLIENT_SECRET,
            user_agent=REDDIT_USER_AGENT,
        )
        # Test connection
        reddit.read_only = True
        _ = reddit.subreddit("test").display_name
        print("  ✅ Connected to Reddit API")
    except Exception as e:
        print(f"  ❌ Failed to connect to Reddit: {e}")
        return 0

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    all_reviews = []
    seen_ids = set()

    for category, subs in SUBREDDITS.items():
        print(f"\n[Category]: {category.upper()}")

        for sub_name in subs:
            print(f"  Subreddit: r/{sub_name}")
            sub_count = 0

            try:
                subreddit = reddit.subreddit(sub_name)

                for query in SEARCH_QUERIES:
                    try:
                        results = subreddit.search(
                            query, sort="relevance", time_filter="all",
                            limit=MAX_POSTS_PER_SUBREDDIT
                        )

                        for post in results:
                            if post.id in seen_ids:
                                continue
                            seen_ids.add(post.id)

                            if not post.is_self or not post.selftext:
                                continue
                            if len(post.selftext) < MIN_BODY_LENGTH:
                                continue
                            if post.score < 2:
                                continue
                            if post.removed_by_category:
                                continue

                            created = datetime.fromtimestamp(
                                post.created_utc, tz=timezone.utc
                            ).strftime("%Y-%m-%d")

                            product_name = extract_product_name(post.title)
                            combined_text = post.title + " " + post.selftext
                            star_rating = derive_star_rating(combined_text)

                            all_reviews.append({
                                "product_name": product_name,
                                "category": category,
                                "product_price": "",
                                "review_title": post.title[:200],
                                "review_text": post.selftext[:5000],
                                "star_rating": star_rating,
                                "review_date": created,
                                "reviewer_name": str(post.author) if post.author else "",
                                "source": f"r/{sub_name}",
                                "reddit_score": post.score,
                            })
                            sub_count += 1

                    except Exception as e:
                        print(f"    ⚠ Search error for '{query}': {e}")
                        continue

            except Exception as e:
                print(f"    ⚠ Error accessing r/{sub_name}: {e}")
                continue

            print(f"    → Collected {sub_count} reviews from r/{sub_name}")

    # Save results
    print(f"\n{'-' * 60}")
    print(f"  Total unique reviews collected: {len(all_reviews)}")

    if all_reviews:
        df = pd.DataFrame(all_reviews)
        df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
        print(f"  ✅ Saved {len(df)} reviews to {OUTPUT_FILE}")
        print(f"  Categories: {df['category'].value_counts().to_dict()}")
        print(f"  Rating distribution: {df['star_rating'].value_counts().sort_index().to_dict()}")
    else:
        print("  ⚠ No reviews collected.")

    print(f"\n  Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    return len(all_reviews)


if __name__ == "__main__":
    scrape_reddit_reviews()
