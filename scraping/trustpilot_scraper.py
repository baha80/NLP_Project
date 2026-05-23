"""
trustpilot_scraper.py
---------------------
Scrape reviews from Trustpilot using BeautifulSoup.
"""

import time
import random
import requests
import pandas as pd
from bs4 import BeautifulSoup
from typing import List, Optional
from datetime import datetime


class TrustpilotScraper:

    BASE_URL = "https://www.trustpilot.com/review/{company}"

    HEADERS_LIST = [
        {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            "Accept-Language": "en-US,en;q=0.9",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
        },
        {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/605.1.15 (KHTML, like Gecko) "
                "Version/17.0 Safari/605.1.15"
            ),
            "Accept-Language": "en-GB,en;q=0.9",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        },
    ]

    def __init__(self, delay: float = 2.0, random_delay: bool = True):
        self.delay        = delay
        self.random_delay = random_delay
        self.session      = requests.Session()

    def _get_headers(self) -> dict:
        return random.choice(self.HEADERS_LIST)

    def _sleep(self):
        d = self.delay
        if self.random_delay:
            d += random.uniform(0.5, 2.0)
        time.sleep(max(d, 1.0))

    def _fetch_page(self, company: str, page: int = 1) -> Optional[BeautifulSoup]:
        url = self.BASE_URL.format(company=company)
        params = {"page": page, "languages": "en"}
        try:
            resp = self.session.get(
                url,
                headers=self._get_headers(),
                params=params,
                timeout=15,
            )
            if resp.status_code == 403:
                print(f"  ⚠️  Blocked on page {page} for {company}.")
                return None
            if resp.status_code == 404:
                print(f"  ❌ '{company}' not found on Trustpilot.")
                return None
            resp.raise_for_status()
            return BeautifulSoup(resp.text, "html.parser")
        except requests.RequestException as e:
            print(f"  Error: {e}")
            return None

    def _parse_reviews(self, soup: BeautifulSoup, company: str) -> List[dict]:
        reviews = []

        # Try multiple selectors for robustness
        cards = (
            soup.select("div[data-service-review-card-paper]") or
            soup.select("article.review") or
            soup.find_all("article", attrs={"data-service-review-card-paper": True})
        )

        for card in cards:
            try:
                # Rating
                rating = None
                rating_el = card.select_one("[data-service-review-rating]")
                if rating_el:
                    try:
                        rating = int(rating_el.get("data-service-review-rating", 0))
                    except (ValueError, TypeError):
                        pass

                # Text
                text = ""
                for sel in [
                    "p[data-service-review-text-typography]",
                    ".review-content__text",
                    "p.typography_body-l__KUYFJ",
                ]:
                    el = card.select_one(sel)
                    if el:
                        text = el.get_text(strip=True)
                        break

                # Title
                title = ""
                for sel in [
                    "h2[data-service-review-title-typography]",
                    ".review-content__title",
                    "h2.typography_heading-s__f7029",
                ]:
                    el = card.select_one(sel)
                    if el:
                        title = el.get_text(strip=True)
                        break

                # Date
                date = ""
                date_el = card.select_one("time")
                if date_el:
                    dt_str = date_el.get("datetime", "")
                    try:
                        date = datetime.fromisoformat(dt_str[:10]).strftime("%Y-%m-%d")
                    except (ValueError, TypeError):
                        date = dt_str[:10]

                # Author
                author = "Anonymous"
                for sel in [
                    "span[data-consumer-name-typography]",
                    ".consumer-information__name",
                    "span.typography_heading-xxs__QKBS8",
                ]:
                    el = card.select_one(sel)
                    if el:
                        author = el.get_text(strip=True)
                        break

                if text and len(text) > 20:
                    reviews.append({
                        "text":      text,
                        "title":     title,
                        "rating":    rating,
                        "date":      date,
                        "user_name": author,
                        "company":   company,
                        "platform":  "Trustpilot",
                    })
            except Exception:
                continue

        return reviews

    def _get_total_pages(self, soup: BeautifulSoup) -> int:
        try:
            nav_buttons = soup.select("nav[aria-label='Pagination'] a")
            if nav_buttons:
                pages = []
                for btn in nav_buttons:
                    try:
                        pages.append(int(btn.get_text(strip=True)))
                    except (ValueError, TypeError):
                        pass
                return max(pages) if pages else 1
        except Exception:
            pass
        return 1

    def collect(self, company: str, max_pages: int = 5) -> pd.DataFrame:
        all_reviews = []
        print(f"🔍 Scraping Trustpilot: {company}")

        first_page = self._fetch_page(company, 1)
        if first_page is None:
            return pd.DataFrame()

        total_pages  = self._get_total_pages(first_page)
        pages_to_get = min(max_pages, total_pages)
        print(f"  Found {total_pages} pages → scraping {pages_to_get}")

        reviews = self._parse_reviews(first_page, company)
        all_reviews.extend(reviews)
        print(f"  Page 1: {len(reviews)} reviews")

        for page in range(2, pages_to_get + 1):
            self._sleep()
            soup = self._fetch_page(company, page)
            if soup is None:
                break
            page_reviews = self._parse_reviews(soup, company)
            all_reviews.extend(page_reviews)
            print(f"  Page {page}: {len(page_reviews)} reviews")

        df = pd.DataFrame(all_reviews)
        if len(df) > 0:
            df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)

        print(f"  ✅ Total: {len(df)} reviews")
        return df

    def collect_multiple(self, companies: List[str], max_pages: int = 3) -> pd.DataFrame:
        all_dfs = []
        for company in companies:
            df = self.collect(company, max_pages=max_pages)
            if len(df) > 0:
                all_dfs.append(df)
            print()
        if not all_dfs:
            return pd.DataFrame()
        combined = pd.concat(all_dfs, ignore_index=True)
        return combined.drop_duplicates(subset=["text"]).reset_index(drop=True)


# ── Companies that work well on Trustpilot ──────────────────────────────────
# Fashion:     shein.com, zara.com, hm.com, asos.com, boohoo.com
# Hotels:      booking.com, expedia.com, hotels.com
# Electronics: samsung.com, apple.com, sony.com
# E-commerce:  ebay.com, etsy.com, aliexpress.com
# Airlines:    turkishairlines.com, ryanair.com

if __name__ == "__main__":
    import os

    scraper = TrustpilotScraper(delay=2.0)

    # Scraping fashion companies — good fit for clothing reviews
    df = scraper.collect_multiple(
        companies=["shein.com", "asos.com", "boohoo.com"],
        max_pages=5,
    )

    if len(df) > 0:
        output_path = os.path.join(os.path.dirname(__file__), "..", "data", "trustpilot_reviews.csv")
        output_path = os.path.normpath(output_path)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"\n✅ Saved {len(df)} reviews → {output_path}")
        print(f"\nBy company:\n{df['company'].value_counts().to_string()}")
        print(f"\nSample:\n{df[['text', 'rating', 'company']].head(5).to_string()}")
    else:
        print("\n⚠️  0 reviews collected.")