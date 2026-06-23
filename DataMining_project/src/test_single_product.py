"""
Quick Single-Product Scraping Test
====================================
Tests the scraper on ONE product from CEL.ro.
Steps:
  1. Fetches the phones listing page and grabs the first product URL
  2. Scrapes that product for reviews
  3. Prints everything it found (name, price, review count, review samples)
  4. Also dumps the raw HTML selectors it tried, so you can debug mismatches

Run:
    python src/test_single_product.py
"""

import requests
from bs4 import BeautifulSoup
import json
import re
import sys
import random

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

BASE_URL = "https://www.cel.ro"

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:128.0) "
    "Gecko/20100101 Firefox/128.0",
]


def headers():
    return {
        "User-Agent": random.choice(USER_AGENTS),
        "Accept-Language": "ro-RO,ro;q=0.9,en-US;q=0.8,en;q=0.7",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }


# ── Step 1: grab first N product URLs from listing page ───────────────────────
def get_product_urls(session, n=10):
    listing_url = f"{BASE_URL}/telefoane-mobile/"
    print(f"\n[1] Fetching listing page: {listing_url}")
    try:
        resp = session.get(listing_url, headers=headers(), timeout=15)
        print(f"    HTTP status: {resp.status_code}")
    except Exception as e:
        print(f"    ERROR: {e}")
        return []

    if resp.status_code != 200:
        print("    Could not load listing page.")
        return []

    soup = BeautifulSoup(resp.text, "html.parser")

    cards = soup.select("a.product_link")
    print(f"    'a.product_link' found: {len(cards)} links")
    if not cards:
        cards = soup.select("div.product_data a[href*='-p']")
        print(f"    fallback selector found: {len(cards)} links")

    urls = []
    seen = set()
    for card in cards:
        href = card.get("href", "")
        if href and ("-l/" in href or "-p" in href):
            full = href if href.startswith("http") else BASE_URL + href
            if full not in seen:
                seen.add(full)
                urls.append(full)
        if len(urls) >= n:
            break

    print(f"    Collected {len(urls)} unique product URLs")
    return urls


# ── Step 2: quick-check a single product, return summary dict ─────────────────
def check_product(session, product_url, idx):
    import time
    print(f"\n{'─'*60}")
    print(f"  [{idx}] {product_url}")

    try:
        resp = session.get(product_url, headers=headers(), timeout=15)
        status = resp.status_code
    except Exception as e:
        print(f"      ERROR: {e}")
        return {"url": product_url, "status": "error", "name": "", "reviews": 0,
                "rating_count": 0, "has_no_reviews_div": False}

    if status != 200:
        print(f"      HTTP {status} — skipping")
        return {"url": product_url, "status": status, "name": "", "reviews": 0,
                "rating_count": 0, "has_no_reviews_div": False}

    soup = BeautifulSoup(resp.text, "html.parser")

    # JSON-LD
    name, rating_count, jsonld_reviews = "", 0, []
    for script in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(script.string)
            items = data if isinstance(data, list) else [data]
            for item in items:
                if item.get("@type") == "Product":
                    name = item.get("name", "")
                    agg = item.get("aggregateRating", {})
                    rating_count = int(agg.get("ratingCount", 0)) if agg else 0
                    for rev in item.get("review", []):
                        text = rev.get("reviewBody", rev.get("description", ""))
                        if text and len(text) > 10:
                            jsonld_reviews.append(text)
        except Exception:
            pass

    has_no_reviews_div = soup.select_one("div.noReviews") is not None

    # HTML review containers
    review_section = soup.select_one('div.productCol.reviews, div[data-tab="reviews"]')
    html_count = 0
    if review_section:
        containers = review_section.select(
            "div.review-item, div.comment-item, div.review, "
            "div[class*='comment'], div[class*='feedback']"
        )
        html_count = len(containers)

    total = len(jsonld_reviews) + html_count
    status_emoji = "✅" if total > 0 else "❌"
    print(f"      Name         : {name[:70] or '(unknown)'}")
    print(f"      Rating votes : {rating_count}")
    print(f"      JSON-LD revs : {len(jsonld_reviews)}")
    print(f"      HTML revs    : {html_count}")
    print(f"      noReviews div: {has_no_reviews_div}")
    print(f"      {status_emoji} Total reviews found: {total}")

    if jsonld_reviews:
        print(f"\n      Sample review text:")
        print(f"      \"{jsonld_reviews[0][:200]}\"")

    # Polite delay
    time.sleep(random.uniform(1.5, 2.5))

    return {
        "url": product_url,
        "name": name,
        "rating_count": rating_count,
        "jsonld_reviews": len(jsonld_reviews),
        "html_reviews": html_count,
        "total_reviews": total,
        "has_no_reviews_div": has_no_reviews_div,
    }


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    N = 10
    session = requests.Session()

    urls = get_product_urls(session, n=N)
    if not urls:
        print("\n❌ Could not find any product URLs. CEL.ro may be blocking.")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  Scanning {len(urls)} products for reviews ...")
    print(f"{'='*60}")

    results = []
    for i, url in enumerate(urls, 1):
        result = check_product(session, url, i)
        results.append(result)

    # ── Summary table ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    with_reviews = [r for r in results if r.get("total_reviews", 0) > 0]
    without      = [r for r in results if r.get("total_reviews", 0) == 0]

    print(f"  Products checked    : {len(results)}")
    print(f"  ✅ Have reviews     : {len(with_reviews)}")
    print(f"  ❌ No reviews found : {len(without)}")

    if with_reviews:
        print(f"\n  Products WITH reviews:")
        for r in with_reviews:
            print(f"    • {r['name'][:60]}")
            print(f"      votes={r['rating_count']}  jsonld={r['jsonld_reviews']}  html={r['html_reviews']}")
            print(f"      {r['url']}")
    else:
        print("\n  ⚠ None of the 10 products had scraped review text.")
        print("  → CEL.ro likely loads reviews via JavaScript (AJAX).")
        print("  → The current requests+BS4 scraper cannot see them.")
        print("  → A Selenium/Playwright-based approach would be needed.")

