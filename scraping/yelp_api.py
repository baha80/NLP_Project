"""
yelp_api.py
-----------
Collect reviews via the official Yelp Fusion API.

Setup:
  1. Go to https://www.yelp.com/developers/v3/manage_app
  2. Create a free app → copy your API key
  3. Add YELP_API_KEY=your_key to your .env file

Free tier: 500 calls/day
"""

import os
import time
import requests
import pandas as pd
from typing import List, Optional
from dotenv import load_dotenv

load_dotenv()

YELP_API_BASE = "https://api.yelp.com/v3"


class YelpScraper:
    """
    Fetch business reviews using Yelp Fusion API.

    Usage:
        scraper = YelpScraper()                          # reads YELP_API_KEY from .env
        scraper = YelpScraper(api_key="YOUR_KEY_HERE")   # or pass directly

        # Search businesses
        businesses = scraper.search_businesses("restaurants", "Paris, France", limit=5)

        # Get reviews for a business
        reviews = scraper.get_reviews(business_id="WavvLdfdP6g8aZTtbBQHTw")

        # Full pipeline: search + collect all reviews
        df = scraper.collect("coffee shops", "New York", n_businesses=10)
        df.to_csv("yelp_reviews.csv", index=False)
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("YELP_API_KEY", "")
        if not self.api_key:
            raise ValueError(
                "Yelp API key not found.\n"
                "1. Get a free key at https://www.yelp.com/developers/v3/manage_app\n"
                "2. Add YELP_API_KEY=your_key to your .env file"
            )
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "application/json",
        }

    # ── Search ──────────────────────────────────────────────────────────────

    def search_businesses(
        self,
        term: str,
        location: str,
        limit: int = 20,
        sort_by: str = "review_count",   # best_match | rating | review_count | distance
    ) -> List[dict]:
        """
        Search for businesses.

        Returns list of dicts:
            id, name, rating, review_count, address, phone, url
        """
        params = {
            "term":     term,
            "location": location,
            "limit":    min(limit, 50),
            "sort_by":  sort_by,
        }

        resp = requests.get(
            f"{YELP_API_BASE}/businesses/search",
            headers=self.headers,
            params=params,
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()

        businesses = []
        for b in data.get("businesses", []):
            businesses.append({
                "id":           b["id"],
                "name":         b["name"],
                "rating":       b.get("rating"),
                "review_count": b.get("review_count"),
                "address":      ", ".join(b.get("location", {}).get("display_address", [])),
                "phone":        b.get("display_phone", ""),
                "url":          b.get("url", ""),
                "category":     b.get("categories", [{}])[0].get("title", ""),
            })

        print(f"Found {len(businesses)} businesses for '{term}' in {location}")
        return businesses

    # ── Reviews ─────────────────────────────────────────────────────────────

    def get_reviews(self, business_id: str) -> List[dict]:
        """
        Fetch up to 3 reviews for a business (Yelp API limit per call).

        Returns list of dicts:
            text, rating, date, user_name, business_id
        """
        resp = requests.get(
            f"{YELP_API_BASE}/businesses/{business_id}/reviews",
            headers=self.headers,
            params={"limit": 3, "sort_by": "yelp_sort"},
            timeout=10,
        )

        if resp.status_code == 429:
            print("Rate limit hit — waiting 60s...")
            time.sleep(60)
            return self.get_reviews(business_id)

        if resp.status_code != 200:
            print(f"Warning: could not fetch reviews for {business_id} ({resp.status_code})")
            return []

        reviews = []
        for r in resp.json().get("reviews", []):
            reviews.append({
                "text":        r.get("text", "").strip(),
                "rating":      r.get("rating"),
                "date":        r.get("time_created", "")[:10],
                "user_name":   r.get("user", {}).get("name", "Anonymous"),
                "business_id": business_id,
                "platform":    "Yelp",
            })

        return reviews

    # ── Full pipeline ────────────────────────────────────────────────────────

    def collect(
        self,
        term: str,
        location: str,
        n_businesses: int = 10,
        delay: float = 0.5,
    ) -> pd.DataFrame:
        """
        Full pipeline: search businesses → collect reviews → return DataFrame.

        Args:
            term:          Search term (e.g. "restaurants", "hotels", "coffee")
            location:      City or address (e.g. "Tunis, Tunisia")
            n_businesses:  How many businesses to query
            delay:         Seconds to wait between API calls (avoid rate limits)

        Returns:
            DataFrame with columns: text, rating, date, user_name, business_id,
                                    platform, business_name, category
        """
        businesses = self.search_businesses(term, location, limit=n_businesses)
        all_reviews = []

        for i, biz in enumerate(businesses):
            print(f"  [{i+1}/{len(businesses)}] Fetching reviews for: {biz['name']}")
            reviews = self.get_reviews(biz["id"])
            for r in reviews:
                r["business_name"] = biz["name"]
                r["category"]      = biz.get("category", term)
                r["business_url"]  = biz["url"]
            all_reviews.extend(reviews)
            time.sleep(delay)

        df = pd.DataFrame(all_reviews)
        df = df[df["text"].str.strip().str.len() > 20]
        df = df.drop_duplicates(subset=["text"])
        df = df.reset_index(drop=True)

        print(f"\n✅ Collected {len(df)} reviews from {len(businesses)} businesses")
        return df
