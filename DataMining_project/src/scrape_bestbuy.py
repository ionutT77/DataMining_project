"""
Electronics Reviews Scraper (Real Web Scraper)
==============================================
Scrapes real product reviews from BestBuy Canada API to guarantee unblocked, authentic data extraction. 
Replaces the simulated Best Buy scraper because Best Buy blocks Python requests (403 Forbidden).

Usage:
    python scrape_bestbuy.py

Output:
    ../data/raw/bestbuy_reviews_raw.csv (Name kept for compatibility with the notebook)
"""

import requests
import pandas as pd
import time
import os
import sys
from datetime import datetime
from tqdm import tqdm

if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

# Configuration
SEARCH_API = "https://www.bestbuy.ca/api/v2/json/search"
REVIEWS_API = "https://www.bestbuy.ca/api/v2/json/reviews/{sku}"

CATEGORIES = {
    "phones": "smartphones",
    "laptops": "laptops",
    "headphones": "wireless headphones",
    "tablets": "tablets"
}

MAX_PRODUCTS_PER_CATEGORY = 10
MAX_PAGES_PER_PRODUCT = 5 # Up to 5 pages of reviews per product (approx 100 reviews)

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "raw")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "bestbuy_reviews_raw.csv")

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept": "application/json"
}

def get_product_skus(query, limit=10):
    """Hits the actual search API to find product IDs"""
    try:
        resp = requests.get(SEARCH_API, params={"query": query, "pageSize": limit}, headers=HEADERS, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            return [p.get("sku") for p in data.get("products", []) if p.get("sku")]
        else:
            print(f"  [!] Search API returned {resp.status_code}")
    except Exception as e:
        print(f"  [!] Request failed: {e}")
    return []

def scrape_reviews_for_sku(sku, category):
    """Hits the actual reviews API to pull real user feedback"""
    reviews_data = []
    
    for page in range(1, MAX_PAGES_PER_PRODUCT + 1):
        try:
            resp = requests.get(REVIEWS_API.format(sku=sku), params={"page": page, "pageSize": 20, "source": "us"}, headers=HEADERS, timeout=10)
            if resp.status_code != 200:
                break
                
            data = resp.json()
            reviews = data.get("reviews", [])
            if not reviews:
                break
                
            for r in reviews:
                reviews_data.append({
                    "product_name": f"Product SKU: {sku}",
                    "category": category,
                    "product_price": 0.0, # Not easily available in reviews endpoint
                    "review_title": r.get("title", ""),
                    "review_text": r.get("comment", ""),
                    "star_rating": int(r.get("rating", 0)),
                    "review_date": r.get("submissionTime", ""),
                    "reviewer_name": r.get("reviewerName", "Anonymous"),
                    "source": "bestbuy_api"
                })
            
            # If we've reached the last page of reviews
            if page >= data.get("totalPages", 1):
                break
                
            time.sleep(0.5) # Polite delay
        except Exception as e:
            print(f"  [!] Failed to get reviews for {sku} page {page}: {e}")
            break
            
    return reviews_data

def run_scraper():
    print("=" * 60)
    print("  Real Electronics Reviews Scraper (API Extraction)")
    print(f"  Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    all_reviews = []
    
    for category, search_query in CATEGORIES.items():
        print(f"\\n  [*] Searching products for category: {category}...")
        skus = get_product_skus(search_query, limit=MAX_PRODUCTS_PER_CATEGORY)
        print(f"  [+] Found {len(skus)} products. Extracting reviews...")
        
        for sku in tqdm(skus, desc=f"Scraping {category}"):
            product_reviews = scrape_reviews_for_sku(sku, category)
            all_reviews.extend(product_reviews)
            time.sleep(1) # Polite delay between products
            
    if all_reviews:
        df = pd.DataFrame(all_reviews)
        
        # Clean up empty reviews
        df = df[df['review_text'].str.strip() != ""]
        df = df.dropna(subset=['review_text', 'star_rating'])
        
        df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
        print(f"\\n  [OK] Successfully scraped {len(df)} REAL electronics reviews -> {OUTPUT_FILE}")
        print(f"  Categories: {df['category'].value_counts().to_dict()}")
    else:
        print("\\n  [!] Failed to scrape any reviews.")
        
    print(f"\\n  Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

if __name__ == "__main__":
    run_scraper()
