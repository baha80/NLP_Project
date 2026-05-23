"""
amazon_scraper.py
-----------------
Scrape Amazon reviews using undetected-chromedriver to bypass bot detection.

Setup:
    pip install undetected-chromedriver selenium pandas

Usage:
    python amazon_scraper.py
"""

import time
import random
import re
import os
import pandas as pd
from typing import List, Optional


class AmazonScraper:

    ASIN_PATTERNS = [
        r"/dp/([A-Z0-9]{10})",
        r"/gp/product/([A-Z0-9]{10})",
        r"/product/([A-Z0-9]{10})",
        r"asin=([A-Z0-9]{10})",
        r"^([A-Z0-9]{10})$",
    ]

    def __init__(self, headless: bool = False, delay: float = 3.0):
        """
        Args:
            headless: Keep False for Amazon — visible window avoids detection
            delay:    Seconds between pages
        """
        self.headless = headless
        self.delay    = delay
        self._driver  = None

    def _init_driver(self):
        if self._driver is not None:
            return

        try:
            import undetected_chromedriver as uc
            options = uc.ChromeOptions()
            options.add_argument("--window-size=1920,1080")
            options.add_argument("--lang=en-US")
            if self.headless:
                options.add_argument("--headless=new")
            self._driver = uc.Chrome(options=options, version_main=147)
            self._disable_driver_destructor(self._driver)
            print("✅ Undetected Chrome initialized")

        except ImportError:
            # Fallback to regular selenium if uc not installed
            from selenium import webdriver
            from selenium.webdriver.chrome.options import Options
            from selenium.webdriver.chrome.service import Service
            from webdriver_manager.chrome import ChromeDriverManager

            options = Options()
            if self.headless:
                options.add_argument("--headless=new")
            options.add_argument("--window-size=1920,1080")
            options.add_argument("--disable-blink-features=AutomationControlled")
            options.add_experimental_option("excludeSwitches", ["enable-automation"])
            options.add_experimental_option("useAutomationExtension", False)
            options.add_argument(
                "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            )
            service = Service(ChromeDriverManager().install())
            self._driver = webdriver.Chrome(service=service, options=options)
            self._driver.execute_script(
                "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"
            )
            print("✅ Chrome WebDriver initialized (standard)")

        self._driver.implicitly_wait(10)

    def _close_driver(self):
        if self._driver:
            try:
                self._driver.quit()
            except Exception:
                pass
            self._driver = None

    @staticmethod
    def _disable_driver_destructor(driver) -> None:
        driver_class = driver.__class__
        if getattr(driver_class, "__module__", "").startswith("undetected_chromedriver"):
            try:
                driver_class.__del__ = lambda self: None
            except Exception:
                pass

    def _sleep(self):
        time.sleep(self.delay + random.uniform(1.0, 3.0))

    def _is_captcha(self) -> bool:
        source = self._driver.page_source.lower()
        title  = self._driver.title.lower()
        return (
            "captcha" in source
            or "robot" in title
            or "something went wrong" in source
            or "enter the characters" in source
            or "sorry, we just need to make sure" in source
        )

    def _handle_captcha(self, page: int):
        """Pause and let the user solve the CAPTCHA manually."""
        print("\n" + "="*55)
        print("🛑 CAPTCHA DETECTED on page", page)
        print("   A Chrome window is open.")
        print("   Please solve the CAPTCHA in the browser window.")
        print("   Then press ENTER here to continue...")
        print("="*55)
        input("   >>> Press ENTER after solving CAPTCHA: ")
        print("   Continuing...\n")
        time.sleep(2)

    def prepare_manual_session(self, start_url: Optional[str] = None) -> str:
        """Open Chrome, let the user navigate manually, then return the current product URL."""
        self._init_driver()
        target_url = start_url or "https://www.amazon.com/"
        self._driver.get(target_url)

        while True:
            print("\n" + "="*60)
            print("🧭 MANUAL AMAZON SESSION")
            print("   Use the open Chrome window to:")
            print("   1. Solve any CAPTCHA or sign-in prompt")
            print("   2. Open the Amazon product page you want to scrape")
            print("   3. Wait until the product page is fully loaded")
            print("="*60)
            input("   >>> Press ENTER when the product page is ready: ")

            current_url = self._driver.current_url
            try:
                asin = self.extract_asin(current_url)
                print(f"   Using current page ASIN: {asin}")
                return current_url
            except ValueError:
                print(f"   Current page is not a supported Amazon product URL: {current_url}")
                print("   Open the product page in Chrome, then press ENTER again.\n")

    def _parse_reviews(self, asin: str, page: int) -> List[dict]:
        """Load review page and extract all reviews."""
        from selenium.webdriver.common.by import By

        url = (
            f"https://www.amazon.com/product-reviews/{asin}/"
            f"?pageNumber={page}&sortBy=recent&reviewerType=all_reviews"
        )
        self._driver.get(url)
        self._sleep()

        # Handle CAPTCHA
        if self._is_captcha():
            self._handle_captcha(page)
            # Reload after solving
            self._driver.get(url)
            self._sleep()

        # Still blocked?
        if self._is_captcha():
            print(f"  ❌ Still blocked on page {page}. Skipping.")
            return []

        reviews = []
        cards = self._driver.find_elements(
            By.CSS_SELECTOR,
            "div[data-hook='review']"
        )

        if not cards:
            # Try alternative
            cards = self._driver.find_elements(By.CSS_SELECTOR, "div.review")

        for card in cards:
            try:
                r = self._extract(card, asin)
                if r:
                    reviews.append(r)
            except Exception:
                continue

        return reviews

    def _extract(self, card, asin: str) -> Optional[dict]:
        from selenium.webdriver.common.by import By

        # Rating
        rating = None
        try:
            el = card.find_element(By.CSS_SELECTOR, "i[data-hook='review-star-rating'] span, i[data-hook='cmps-review-star-rating'] span")
            rating = float(el.get_attribute("innerHTML").strip().split()[0])
        except Exception:
            pass

        # Title
        title = ""
        try:
            el = card.find_element(By.CSS_SELECTOR, "a[data-hook='review-title'] span:not(.a-icon-alt)")
            title = el.text.strip()
        except Exception:
            pass

        # Body
        text = ""
        try:
            el = card.find_element(By.CSS_SELECTOR, "span[data-hook='review-body'] span")
            text = el.text.strip()
        except Exception:
            pass

        # Date
        date = ""
        try:
            el = card.find_element(By.CSS_SELECTOR, "span[data-hook='review-date']")
            raw = el.text.strip()
            if " on " in raw:
                raw = raw.split(" on ")[-1]
            from datetime import datetime
            for fmt in ["%B %d, %Y", "%d %B %Y"]:
                try:
                    date = datetime.strptime(raw, fmt).strftime("%Y-%m-%d")
                    break
                except ValueError:
                    pass
            if not date:
                date = raw
        except Exception:
            pass

        # Verified
        verified = False
        try:
            card.find_element(By.CSS_SELECTOR, "span[data-hook='avp-badge']")
            verified = True
        except Exception:
            pass

        if not text or len(text) < 10:
            return None

        return {
            "text":     text,
            "title":    title,
            "rating":   rating,
            "date":     date,
            "asin":     asin,
            "verified": verified,
            "platform": "Amazon",
        }

    def collect_by_asin(
        self,
        asin: str,
        max_pages: int = 10,
        product_name: Optional[str] = None,
    ) -> pd.DataFrame:

        self._init_driver()
        all_reviews = []

        try:
            print(f"🔍 Scraping Amazon — ASIN: {asin}")
            for page in range(1, max_pages + 1):
                reviews = self._parse_reviews(asin, page)
                if not reviews:
                    print(f"  Page {page}: no reviews — stopping")
                    break
                all_reviews.extend(reviews)
                print(f"  Page {page}: {len(reviews)} reviews (total: {len(all_reviews)})")
        finally:
            self._close_driver()

        df = pd.DataFrame(all_reviews)
        if len(df) > 0:
            if product_name:
                df["product"] = product_name
            df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)

        print(f"✅ Total: {len(df)} reviews collected")
        return df

    @classmethod
    def extract_asin(cls, product_ref: str) -> str:
        product_ref = str(product_ref).strip()
        for pattern in cls.ASIN_PATTERNS:
            match = re.search(pattern, product_ref, flags=re.IGNORECASE)
            if match:
                return match.group(1).upper()
        raise ValueError(f"Could not extract ASIN from input: {product_ref}")

    def collect(self, product_url: str, max_pages: int = 10, product_name: Optional[str] = None) -> pd.DataFrame:
        asin = self.extract_asin(product_url)
        return self.collect_by_asin(asin, max_pages=max_pages, product_name=product_name)

    def collect_multiple(
        self,
        products: List[str],
        max_pages: int = 5,
        names: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        all_frames = []
        names = names or []

        for index, product_ref in enumerate(products):
            product_name = names[index] if index < len(names) else None
            try:
                df = self.collect(product_ref, max_pages=max_pages, product_name=product_name)
                if not df.empty:
                    all_frames.append(df)
            except Exception as exc:
                print(f"❌ Amazon product skipped ({product_ref}): {exc}")

        if not all_frames:
            return pd.DataFrame()

        return pd.concat(all_frames, ignore_index=True).drop_duplicates(subset=["text", "asin"]).reset_index(drop=True)


if __name__ == "__main__":

    scraper = AmazonScraper(headless=False)

    df = scraper.collect(
        product_url="https://www.amazon.com/dp/B0F1CMH5NN",
        max_pages=5,
        product_name="WIHOLL Dresses",
    )

    if len(df) > 0:
        output_path = os.path.join(os.path.dirname(__file__), "..", "data", "amazon_reviews.csv")
        output_path = os.path.normpath(output_path)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"\n✅ Saved {len(df)} reviews → {output_path}")
        print(df[["text", "rating", "date"]].head(10).to_string())
    else:
        print("\n⚠️  0 reviews collected.")
