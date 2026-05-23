"""
scraper_pipeline.py
-------------------
Unified pipeline that combines all scrapers, deduplicates,
normalizes columns, and saves to CSV.

Usage:
    pipeline = ScraperPipeline(output_dir="data")

    # Collect from all sources
    df = pipeline.run(
        yelp_term="restaurants",
        yelp_location="Tunis, Tunisia",
        trustpilot_companies=["booking.com", "tripadvisor.com"],
        amazon_asins=["B08N5WRWNW"],
        max_pages=3,
    )

    df.to_csv("data/all_reviews.csv", index=False)
"""

import os
import pandas as pd
from typing import List, Optional
from datetime import datetime


class ScraperPipeline:
    """
    Orchestrates multiple scrapers and produces a unified dataset.

    Output columns (standardized):
        text, rating, date, platform, category, source_id
    """

    REQUIRED_COLS = ["text", "rating", "date", "platform"]

    def __init__(self, output_dir: str = "data"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    # ── Individual source runners ────────────────────────────────────────────

    def _run_yelp(
        self,
        term: str,
        location: str,
        n_businesses: int = 10,
    ) -> pd.DataFrame:
        try:
            from scraping.yelp_api import YelpScraper
            scraper = YelpScraper()
            df = scraper.collect(term, location, n_businesses=n_businesses)
            return self._normalize(df, default_category=term)
        except ValueError as e:
            print(f"⚠️  Yelp skipped: {e}")
            return pd.DataFrame()
        except Exception as e:
            print(f"❌ Yelp error: {e}")
            return pd.DataFrame()

    def _run_trustpilot(
        self,
        companies: List[str],
        max_pages: int = 3,
    ) -> pd.DataFrame:
        try:
            from scraping.trustpilot_scraper import TrustpilotScraper
            scraper = TrustpilotScraper()
            df = scraper.collect_multiple(companies, max_pages=max_pages)
            return self._normalize(df, default_category="E-commerce")
        except Exception as e:
            print(f"❌ Trustpilot error: {e}")
            return pd.DataFrame()

    def _run_tripadvisor(
        self,
        urls: List[str],
        names: Optional[List[str]],
        max_pages: int = 3,
    ) -> pd.DataFrame:
        try:
            from scraping.tripadvisor_scraper import TripAdvisorScraper
            scraper = TripAdvisorScraper(headless=True)
            df = scraper.collect_multiple(urls, max_pages=max_pages, names=names)
            return self._normalize(df, default_category="Hotel/Restaurant")
        except Exception as e:
            print(f"❌ TripAdvisor error: {e}")
            return pd.DataFrame()

    def _run_amazon(
        self,
        asins: List[str],
        names: Optional[List[str]],
        max_pages: int = 5,
    ) -> pd.DataFrame:
        try:
            from scraping.amazon_scraper import AmazonScraper
            scraper = AmazonScraper(headless=True)
            df = scraper.collect_multiple(asins, max_pages=max_pages, names=names)
            return self._normalize(df, default_category="Product")
        except Exception as e:
            print(f"❌ Amazon error: {e}")
            return pd.DataFrame()

    # ── Normalization ────────────────────────────────────────────────────────

    def _normalize(self, df: pd.DataFrame, default_category: str = "General") -> pd.DataFrame:
        """Ensure all DataFrames have the same structure."""
        if df.empty:
            return df

        # Required columns with defaults
        if "text" not in df.columns:
            return pd.DataFrame()
        if "rating" not in df.columns:
            df["rating"] = None
        if "date" not in df.columns:
            df["date"] = datetime.today().strftime("%Y-%m-%d")
        if "platform" not in df.columns:
            df["platform"] = "Unknown"
        if "category" not in df.columns:
            df["category"] = default_category

        # Clean text
        df["text"] = df["text"].astype(str).str.strip()
        df = df[df["text"].str.len() > 20]

        # Normalize rating to 1-5
        if df["rating"].notna().any():
            df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
            df["rating"] = df["rating"].clip(1, 5)

        # Keep only needed columns + extras
        keep = ["text", "rating", "date", "platform", "category"]
        extra = [c for c in ["title", "user_name", "business_name", "product", "company", "asin"] if c in df.columns]
        return df[keep + extra].reset_index(drop=True)

    # ── Main pipeline ─────────────────────────────────────────────────────────

    def run(
        self,
        # Yelp
        yelp_term: Optional[str] = None,
        yelp_location: Optional[str] = None,
        yelp_n_businesses: int = 10,
        # Trustpilot
        trustpilot_companies: Optional[List[str]] = None,
        trustpilot_max_pages: int = 3,
        # TripAdvisor
        tripadvisor_urls: Optional[List[str]] = None,
        tripadvisor_names: Optional[List[str]] = None,
        tripadvisor_max_pages: int = 3,
        # Amazon
        amazon_asins: Optional[List[str]] = None,
        amazon_names: Optional[List[str]] = None,
        amazon_max_pages: int = 5,
        # Output
        save_csv: bool = True,
        filename: str = "scraped_reviews.csv",
    ) -> pd.DataFrame:
        """
        Run the full scraping pipeline.

        Returns:
            Unified, deduplicated DataFrame ready for NLP analysis.
        """
        all_dfs = []

        # ── Yelp ──────────────────────────────────────────────────────────────
        if yelp_term and yelp_location:
            print("\n" + "="*50)
            print("📍 YELP")
            print("="*50)
            df = self._run_yelp(yelp_term, yelp_location, yelp_n_businesses)
            if not df.empty:
                all_dfs.append(df)
                print(f"  → {len(df)} Yelp reviews collected")

        # ── Trustpilot ────────────────────────────────────────────────────────
        if trustpilot_companies:
            print("\n" + "="*50)
            print("⭐ TRUSTPILOT")
            print("="*50)
            df = self._run_trustpilot(trustpilot_companies, trustpilot_max_pages)
            if not df.empty:
                all_dfs.append(df)
                print(f"  → {len(df)} Trustpilot reviews collected")

        # ── TripAdvisor ───────────────────────────────────────────────────────
        if tripadvisor_urls:
            print("\n" + "="*50)
            print("✈️  TRIPADVISOR")
            print("="*50)
            df = self._run_tripadvisor(tripadvisor_urls, tripadvisor_names, tripadvisor_max_pages)
            if not df.empty:
                all_dfs.append(df)
                print(f"  → {len(df)} TripAdvisor reviews collected")

        # ── Amazon ────────────────────────────────────────────────────────────
        if amazon_asins:
            print("\n" + "="*50)
            print("🛒 AMAZON")
            print("="*50)
            df = self._run_amazon(amazon_asins, amazon_names, amazon_max_pages)
            if not df.empty:
                all_dfs.append(df)
                print(f"  → {len(df)} Amazon reviews collected")

        # ── Combine ───────────────────────────────────────────────────────────
        if not all_dfs:
            print("\n⚠️ No data collected. Check your configuration.")
            return pd.DataFrame()

        combined = pd.concat(all_dfs, ignore_index=True)
        combined = combined.drop_duplicates(subset=["text"])
        combined = combined.reset_index(drop=True)

        # Add unique ID
        combined.insert(0, "review_id", range(1, len(combined) + 1))

        print("\n" + "="*50)
        print(f"✅ PIPELINE COMPLETE")
        print(f"   Total reviews:  {len(combined)}")
        print(f"   Platforms: {combined['platform'].value_counts().to_dict()}")
        print("="*50)

        if save_csv:
            output_path = os.path.join(self.output_dir, filename)
            combined.to_csv(output_path, index=False)
            print(f"   Saved → {output_path}")

        return combined

    def load_existing(self, filename: str = "scraped_reviews.csv") -> pd.DataFrame:
        """Load a previously scraped CSV."""
        path = os.path.join(self.output_dir, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"No file found at {path}")
        df = pd.read_csv(path)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
        print(f"Loaded {len(df)} reviews from {path}")
        return df
