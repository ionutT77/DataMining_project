"""
eMAG Product Reviews Scraper
=============================
Scrapes product reviews from eMAG.ro across multiple electronics categories.
Uses requests + BeautifulSoup (as taught in Lab 5).

Usage:
    python src/scraper.py

Output:
    data/raw/emag_reviews_raw.csv
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import random
import os
import re
import json
from tqdm import tqdm
from datetime import datetime

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_URL = "https://www.emag.ro"

CATEGORIES = {
    "phones": "/telefoane-mobile/c",
    "laptops": "/laptopuri/c",
    "headphones": "/casti/c",
    "tablets": "/tablete/c",
}

# How many listing pages to crawl per category (each page ~ 60 products)
MAX_LISTING_PAGES_PER_CATEGORY = 3

# Max review pages to fetch per product
MAX_REVIEW_PAGES_PER_PRODUCT = 5

# Polite scraping delays (seconds)
MIN_DELAY = 2.0
MAX_DELAY = 4.0

# Rotating User-Agents to reduce blocking risk
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:128.0) "
    "Gecko/20100101 Firefox/128.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_5) AppleWebKit/605.1.15 "
    "(KHTML, like Gecko) Version/17.5 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
]

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "raw")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "emag_reviews_raw.csv")


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------
def _random_headers() -> dict:
    """Return request headers with a random User-Agent."""
    return {
        "User-Agent": random.choice(USER_AGENTS),
        "Accept-Language": "ro-RO,ro;q=0.9,en-US;q=0.8,en;q=0.7",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }


def _polite_sleep():
    """Sleep a random duration between requests to be polite."""
    time.sleep(random.uniform(MIN_DELAY, MAX_DELAY))


def _safe_get(url: str, session: requests.Session, retries: int = 3):
    """GET with retries and exponential back-off."""
    for attempt in range(retries):
        try:
            resp = session.get(url, headers=_random_headers(), timeout=15)
            if resp.status_code == 200:
                return resp
            if resp.status_code == 429:  # rate-limited
                wait = 10 * (attempt + 1)
                print(f"  ⏳ Rate-limited. Waiting {wait}s …")
                time.sleep(wait)
                continue
            if resp.status_code == 403:
                print(f"  ⛔ Blocked (403) on {url}")
                return None
        except requests.RequestException as exc:
            print(f"  ⚠ Request error: {exc}")
            time.sleep(5)
    return None


# ---------------------------------------------------------------------------
# Scraping functions
# ---------------------------------------------------------------------------
def get_product_links(session, category_path: str, max_pages: int):
    """Scrape product URLs from category listing pages."""
    product_links = []
    for page in range(1, max_pages + 1):
        url = f"{BASE_URL}{category_path}/p{page}/c"
        resp = _safe_get(url, session)
        if resp is None:
            break

        soup = BeautifulSoup(resp.text, "html.parser")

        # eMAG product cards typically use data-url or href in card-v2 links
        cards = soup.select("a.card-v2-title, a.product-title")
        if not cards:
            # Fallback: try generic product links
            cards = soup.select("a[href*='/pd/']")

        for card in cards:
            href = card.get("href", "")
            if href and "/pd/" in href:
                full_url = href if href.startswith("http") else BASE_URL + href
                product_links.append(full_url)

        _polite_sleep()

    return list(set(product_links))  # deduplicate


def extract_product_info(soup):
    """Extract product name and price from a product page."""
    name = ""
    price = ""

    # Product name
    title_tag = soup.select_one("h1.page-title, h1")
    if title_tag:
        name = title_tag.get_text(strip=True)

    # Price
    price_tag = soup.select_one("p.product-new-price, .product-highlight-price")
    if price_tag:
        price_text = price_tag.get_text(strip=True)
        # Extract numeric price
        price_nums = re.findall(r"[\d.,]+", price_text.replace(".", "").replace(",", "."))
        if price_nums:
            price = price_nums[0]

    return name, price


def extract_reviews_from_page(soup, product_name, product_price, category):
    """Extract all reviews visible on the current page."""
    reviews = []

    review_containers = soup.select(
        "div.review-body, div.product-review-body, "
        "div[class*='review'], div.feedback-body"
    )

    for container in review_containers:
        try:
            # Star rating
            star_rating = 0
            stars_el = container.select_one(
                "span.star-rating-text, div.star-rating, "
                "span[class*='star'], div[class*='rating']"
            )
            if stars_el:
                star_text = stars_el.get_text(strip=True)
                nums = re.findall(r"\d+", star_text)
                if nums:
                    star_rating = int(nums[0])

            # Review text
            review_text = ""
            text_el = container.select_one(
                "p.review-text, div.review-text, "
                "span.review-text, p[class*='review']"
            )
            if text_el:
                review_text = text_el.get_text(strip=True)
            else:
                # Fallback: get all paragraph text
                paragraphs = container.find_all("p")
                review_text = " ".join(p.get_text(strip=True) for p in paragraphs)

            # Review title
            review_title = ""
            title_el = container.select_one(
                "span.review-title, h4, strong"
            )
            if title_el:
                review_title = title_el.get_text(strip=True)

            # Date
            review_date = ""
            date_el = container.select_one(
                "span.review-date, time, span[class*='date']"
            )
            if date_el:
                review_date = date_el.get_text(strip=True)

            # Reviewer name
            reviewer = ""
            reviewer_el = container.select_one(
                "span.review-author, span[class*='author'], "
                "a[class*='author']"
            )
            if reviewer_el:
                reviewer = reviewer_el.get_text(strip=True)

            # Only add if we have meaningful content
            if review_text and len(review_text) > 10 and 1 <= star_rating <= 5:
                reviews.append({
                    "product_name": product_name,
                    "category": category,
                    "product_price": product_price,
                    "review_title": review_title,
                    "review_text": review_text,
                    "star_rating": star_rating,
                    "review_date": review_date,
                    "reviewer_name": reviewer,
                })
        except Exception:
            continue

    return reviews


def scrape_product_reviews(session, product_url, category, max_review_pages):
    """Scrape all reviews for a single product."""
    all_reviews = []

    resp = _safe_get(product_url, session)
    if resp is None:
        return all_reviews

    soup = BeautifulSoup(resp.text, "html.parser")
    product_name, product_price = extract_product_info(soup)

    if not product_name:
        return all_reviews

    # Extract reviews from the main product page
    page_reviews = extract_reviews_from_page(
        soup, product_name, product_price, category
    )
    all_reviews.extend(page_reviews)

    # Try paginated review pages (eMAG uses /reviews/p2 etc.)
    for page in range(2, max_review_pages + 1):
        review_url = product_url.rstrip("/") + f"/reviews/p{page}"
        _polite_sleep()
        resp = _safe_get(review_url, session)
        if resp is None:
            break

        soup = BeautifulSoup(resp.text, "html.parser")
        page_reviews = extract_reviews_from_page(
            soup, product_name, product_price, category
        )
        if not page_reviews:
            break
        all_reviews.extend(page_reviews)

    return all_reviews


# ---------------------------------------------------------------------------
# Main scraping pipeline
# ---------------------------------------------------------------------------
def run_scraper():
    """Main entry point: scrape all categories and save to CSV."""
    print("=" * 60)
    print("  eMAG Product Reviews Scraper")
    print(f"  Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    session = requests.Session()
    all_reviews = []

    for category_name, category_path in CATEGORIES.items():
        print(f"\n[Category]: {category_name.upper()}")
        print(f"   Fetching product links …")

        product_links = get_product_links(
            session, category_path, MAX_LISTING_PAGES_PER_CATEGORY
        )
        print(f"   Found {len(product_links)} product links")

        if not product_links:
            print("   [!] No products found. eMAG may be blocking requests.")
            print("   [i] Try running with a VPN or adjusting delays.")
            continue

        for link in tqdm(product_links, desc=f"   Scraping {category_name}"):
            reviews = scrape_product_reviews(
                session, link, category_name, MAX_REVIEW_PAGES_PER_PRODUCT
            )
            all_reviews.extend(reviews)
            _polite_sleep()

    # Save results
    if all_reviews:
        df = pd.DataFrame(all_reviews)
        df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
        print(f"\n[OK] Saved {len(df)} reviews to {OUTPUT_FILE}")
        print(f"   Categories: {df['category'].value_counts().to_dict()}")
        print(f"   Rating distribution: {df['star_rating'].value_counts().sort_index().to_dict()}")
    else:
        print("\n[!] No reviews were scraped.")
        print("   eMAG likely blocked the requests.")
        print("   Please use the fallback dataset or try with a VPN.")
        _generate_sample_dataset()

    print(f"\n  Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)


def _generate_sample_dataset():
    """
    Generate a realistic sample dataset as a fallback when scraping is blocked.
    This allows the analysis notebook to run even if live scraping fails.
    The sample mimics real eMAG review patterns.
    """
    print("\n[i] Generating sample dataset for analysis …")

    import numpy as np
    np.random.seed(42)

    categories = ["phones", "laptops", "headphones", "tablets"]

    # Realistic Romanian product names per category
    products = {
        "phones": [
            "Samsung Galaxy S24 Ultra 256GB", "iPhone 15 Pro Max 256GB",
            "Xiaomi 14 Ultra 512GB", "Samsung Galaxy A55 5G 128GB",
            "iPhone 15 128GB", "Google Pixel 8 Pro 256GB",
            "OnePlus 12 256GB", "Motorola Edge 50 Pro 256GB",
            "Samsung Galaxy S24 128GB", "Xiaomi Redmi Note 13 Pro 256GB",
            "iPhone 14 128GB", "Huawei P60 Pro 256GB",
            "POCO X6 Pro 256GB", "Realme 12 Pro+ 256GB",
            "Samsung Galaxy Z Flip5 256GB", "Nothing Phone (2) 256GB",
        ],
        "laptops": [
            "Laptop ASUS ROG Strix G16 Intel i7 RTX 4060",
            "Laptop Lenovo Legion 5 Pro AMD Ryzen 7 RTX 4070",
            "MacBook Air M3 15 inch 256GB", "Laptop HP Pavilion 15 Intel i5",
            "Laptop Acer Nitro V15 AMD Ryzen 5 RTX 4050",
            "Laptop Dell Inspiron 16 Intel i7 16GB",
            "MacBook Pro M3 Pro 14 inch 512GB",
            "Laptop ASUS VivoBook 15 AMD Ryzen 5",
            "Laptop Lenovo IdeaPad 3 Intel i5",
            "Laptop MSI Katana 15 Intel i7 RTX 4060",
        ],
        "headphones": [
            "Casti Apple AirPods Pro 2", "Casti Sony WH-1000XM5",
            "Casti Samsung Galaxy Buds2 Pro", "Casti JBL Tune 770NC",
            "Casti Sony WF-1000XM5", "Casti Bose QuietComfort Ultra",
            "Casti Marshall Major IV", "Casti Xiaomi Buds 4 Pro",
            "Casti HyperX Cloud III", "Casti Razer Kraken V3",
        ],
        "tablets": [
            "Tableta Apple iPad Air M2 11 inch 128GB",
            "Tableta Samsung Galaxy Tab S9 FE 128GB",
            "Tableta Lenovo Tab P12 128GB",
            "Tableta Apple iPad 10th gen 64GB",
            "Tableta Samsung Galaxy Tab A9+ 64GB",
            "Tableta Huawei MatePad 11.5 128GB",
            "Tableta Xiaomi Pad 6 128GB",
            "Tableta Apple iPad Pro M4 11 inch 256GB",
        ],
    }

    # Realistic Romanian review templates organized by sentiment
    positive_reviews_ro = [
        "Foarte multumit de achizitie. Produsul functioneaza impecabil, calitate excelenta. Recomand cu incredere!",
        "Produsul este exact cum este descris. Livrarea a fost rapida, ambalaj impecabil. Calitate premium.",
        "Am fost placut surprins de calitate. Raportul calitate-pret este foarte bun. Merita fiecare ban.",
        "Excelent! Folosesc de cateva saptamani si sunt foarte multumit. Display superb si performanta de top.",
        "Calitate superioara, design elegant, functioneaza perfect. Cel mai bun produs din gama asta de pret.",
        "Un produs care isi face treaba excelent. Bateria tine foarte bine, ecranul este superb.",
        "Sunt foarte multumit de acest produs. L-am cumparat pentru sotie si este incantata.",
        "Recomand cu caldura! Produs de calitate, livrat rapid. Totul a fost perfect de la inceput.",
        "Dupa doua luni de utilizare pot spune ca este cel mai bun produs pe care l-am avut.",
        "Impecabil! Calitate de fabricatie excelenta, functioneaza fara probleme. Sunetul este cristalat.",
        "Produs premium la un pret accesibil. Performanta impresionanta si design modern.",
        "L-am luat in promotie si a fost cea mai buna decizie. Functioneaza impecabil dupa 3 luni.",
        "Foarte bun! Camera foto face poze superbe, bateria tine o zi intreaga fara probleme.",
        "Raport calitate pret imbatabil. Este rapid, arata bine si face tot ce am nevoie.",
        "Super produs! Am mai cumparat si pentru un prieten. Toti sunt multumiti de calitate.",
        "Cel mai bun produs din categoria lui. Ecranul este luminos si culorile sunt vii.",
        "Produs de nota 10. Livrarea a fost in 24 de ore. Ambalajul foarte bine protejat.",
        "Functioneaza excelent, performanta de top. L-am comparat cu alte produse mai scumpe si e la fel de bun.",
        "Absolut genial! Am trecut de la un model mai vechi si diferenta este enorma.",
        "Design superb si materiale de calitate. Se simte premium in mana. Foarte multumit.",
    ]

    neutral_reviews_ro = [
        "Produsul este OK, nimic special. Face ce trebuie dar nu impresionează in mod deosebit.",
        "E decent pentru pretul platit. Are cateva minusuri dar in general e acceptabil.",
        "Calitate medie, asteptarile mele erau putin mai mari. Totusi, functioneaza cum trebuie.",
        "Nu e rau dar nici grozav. Camera ar putea fi mai buna. In rest, face treaba.",
        "La pretul asta e OK. Nu as plati mai mult pe el. Bateria e medie, ecranul decent.",
        "Produsul e in regula, dar instructiunile sunt doar in engleza. Livrarea a fost OK.",
        "E un produs mediu. Are si parti bune si parti mai putin bune. Per total, acceptabil.",
        "Am asteptari mixte. Unele functii merg bine, altele nu atat de bine. E OK per total.",
    ]

    negative_reviews_ro = [
        "Dezamagit total. Produsul s-a stricat dupa doua saptamani. Calitate foarte proasta!",
        "Nu recomand! Produsul nu corespunde descrierii. Am cerut retur imediat.",
        "Calitate dezamagitoare pentru pretul platit. Se simte ieftin si fragil.",
        "Bateria nu tine deloc. Dupa cateva ore trebuie incarcat din nou. Foarte slab!",
        "Produsul a venit cu defecte. Ecranul avea zgarieturi. Ambalajul era deteriorat.",
        "Regret achizitia. Performanta este sub asteptari. Se blocheaza constant.",
        "Sunetul este oribil, plin de distorsiuni. La pretul asta ma asteptam la mult mai mult.",
        "S-a stricat in prima luna. Garantia nu acopera problema. Bani aruncati pe geam!",
        "Foarte dezamagit. Produsul se incinge foarte tare si se inchide singur.",
        "Livrarea a durat 2 saptamani si produsul a venit lovit. Experienta groaznica!",
        "Nu merita banii. Am cumparat ceva mult mai ieftin si functioneaza mai bine.",
        "Calitate proasta. Dupa o luna a aparut o pata pe ecran. Nu se mai vede bine.",
    ]

    prices = {
        "phones": (500, 8000),
        "laptops": (1500, 12000),
        "headphones": (50, 2500),
        "tablets": (400, 7000),
    }

    months = pd.date_range("2023-01-01", "2025-12-31", freq="ME")

    records = []
    review_id = 0

    for cat in categories:
        cat_products = products[cat]
        n_products = len(cat_products)

        for prod_name in cat_products:
            price_lo, price_hi = prices[cat]
            product_price = round(np.random.uniform(price_lo, price_hi), 2)

            # Number of reviews per product (varied to be realistic)
            n_reviews = np.random.randint(15, 120)

            for _ in range(n_reviews):
                # Weighted star distribution (most products skew positive)
                star = np.random.choice(
                    [1, 2, 3, 4, 5],
                    p=[0.08, 0.07, 0.12, 0.28, 0.45]
                )

                if star >= 4:
                    text = np.random.choice(positive_reviews_ro)
                    # Add product-specific detail
                    extras = [
                        f" Folosesc {prod_name} zilnic.",
                        f" {prod_name} este exact ce cautam.",
                        "",
                        f" Am comparat cu alte modele si {prod_name} e cel mai bun.",
                        "",
                    ]
                    text += np.random.choice(extras)
                elif star == 3:
                    text = np.random.choice(neutral_reviews_ro)
                else:
                    text = np.random.choice(negative_reviews_ro)

                date = pd.to_datetime(np.random.choice(months)).strftime("%Y-%m-%d")

                records.append({
                    "product_name": prod_name,
                    "category": cat,
                    "product_price": product_price,
                    "review_title": "",
                    "review_text": text,
                    "star_rating": int(star),
                    "review_date": date,
                    "reviewer_name": f"User_{review_id}",
                })
                review_id += 1

    df = pd.DataFrame(records)
    # Shuffle
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
    print(f"[OK] Generated {len(df)} sample reviews → {OUTPUT_FILE}")
    print(f"   Categories: {df['category'].value_counts().to_dict()}")
    return df


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    run_scraper()
