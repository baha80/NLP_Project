"""
tripadvisor_scraper.py
----------------------
Scrape reviews from TripAdvisor using Selenium (headless Chrome).
Selenium is required because TripAdvisor loads content with JavaScript.

Setup:
    pip install selenium webdriver-manager

    Chrome must be installed on your system.
    webdriver-manager auto-downloads the matching ChromeDriver.

Usage:
    scraper = TripAdvisorScraper(headless=True)
    df = scraper.collect(
        url="https://www.tripadvisor.com/Hotel_Review-g60763-d122572-Reviews-The_Plaza-New_York_City_New_York.html",
        max_pages=5
    )
    df.to_csv("data/tripadvisor_reviews.csv", index=False)
"""

import time
import random
import pandas as pd
from typing import List, Optional
from datetime import datetime


class TripAdvisorScraper:
    """
    Selenium-based TripAdvisor review scraper.

    Usage:
        scraper = TripAdvisorScraper(headless=True)

        # Single property
        url = "https://www.tripadvisor.com/Hotel_Review-g60763-d122572-..."
        df  = scraper.collect(url, max_pages=5)

        # Multiple properties
        urls = ["https://...", "https://..."]
        df   = scraper.collect_multiple(urls, max_pages=3)
    """

    def __init__(self, headless: bool = True, delay: float = 2.5):
        """
        Args:
            headless: Run Chrome in background (no window)
            delay:    Seconds between page loads
        """
        self.headless = headless
        self.delay    = delay
        self._driver  = None

    def _init_driver(self):
        """Initialize the Selenium WebDriver (lazy)."""
        if self._driver is not None:
            return

        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options
        from selenium.webdriver.chrome.service import Service
        from webdriver_manager.chrome import ChromeDriverManager

        options = Options()
        if self.headless:
            options.add_argument("--headless=new")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--disable-blink-features=AutomationControlled")
        options.add_experimental_option("excludeSwitches", ["enable-automation"])
        options.add_experimental_option("useAutomationExtension", False)
        options.add_argument(
            "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )

        service = Service(ChromeDriverManager().install())
        self._driver = webdriver.Chrome(service=service, options=options)
        self._driver.implicitly_wait(10)

        # Mask webdriver property
        self._driver.execute_script(
            "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"
        )
        print("✅ Chrome WebDriver initialized")

    def _close_driver(self):
        if self._driver:
            self._driver.quit()
            self._driver = None

    def _sleep(self, extra: float = 0):
        time.sleep(self.delay + random.uniform(0, 1.5) + extra)

    def _expand_reviews(self):
        """Click 'Read more' buttons to expand truncated reviews."""
        from selenium.webdriver.common.by import By

        try:
            more_btns = self._driver.find_elements(
                By.CSS_SELECTOR, "button.taLnk.ulBlueLinks"
            )
            for btn in more_btns[:10]:
                try:
                    self._driver.execute_script("arguments[0].click();", btn)
                    time.sleep(0.3)
                except Exception:
                    pass
        except Exception:
            pass

    def _parse_page(self, url_base: str) -> List[dict]:
        """Extract reviews from the current page."""
        from selenium.webdriver.common.by import By

        reviews = []

        try:
            review_cards = self._driver.find_elements(
                By.CSS_SELECTOR, "div.review-container, div[data-reviewid]"
            )

            if not review_cards:
                # Try alternative selectors
                review_cards = self._driver.find_elements(
                    By.CSS_SELECTOR, "div._c"
                )

            for card in review_cards:
                try:
                    review = self._extract_from_card(card)
                    if review:
                        reviews.append(review)
                except Exception:
                    continue

        except Exception as e:
            print(f"  Parse error: {e}")

        return reviews

    def _extract_from_card(self, card) -> Optional[dict]:
        """Extract data from a single review element."""
        from selenium.webdriver.common.by import By

        # ── Rating ───────────────────────────────────────────────────────────
        rating = None
        try:
            rating_el = card.find_element(By.CSS_SELECTOR, "span.ui_bubble_rating")
            cls = rating_el.get_attribute("class")
            for c in cls.split():
                if "bubble_" in c:
                    rating = int(c.split("_")[1]) // 10
                    break
        except Exception:
            pass

        # ── Text ─────────────────────────────────────────────────────────────
        text = ""
        try:
            text_el = card.find_element(By.CSS_SELECTOR, "p.partial_entry, span.fullText, p.review-full-text")
            text = text_el.text.strip()
        except Exception:
            pass

        # ── Title ─────────────────────────────────────────────────────────────
        title = ""
        try:
            title_el = card.find_element(By.CSS_SELECTOR, "span.noQuotes, a.title")
            title = title_el.text.strip()
        except Exception:
            pass

        # ── Date ──────────────────────────────────────────────────────────────
        date = ""
        try:
            date_el = card.find_element(By.CSS_SELECTOR, "span.ratingDate")
            date = date_el.get_attribute("title") or date_el.text.strip()
            try:
                parsed = datetime.strptime(date, "%B %d, %Y")
                date = parsed.strftime("%Y-%m-%d")
            except ValueError:
                pass
        except Exception:
            pass

        # ── Author ────────────────────────────────────────────────────────────
        author = "Anonymous"
        try:
            author_el = card.find_element(By.CSS_SELECTOR, "div.info_text div")
            author = author_el.text.strip()
        except Exception:
            pass

        if not text or len(text) < 20:
            return None

        return {
            "text":      text,
            "title":     title,
            "rating":    rating,
            "date":      date,
            "user_name": author,
            "platform":  "TripAdvisor",
        }

    def _go_to_next_page(self) -> bool:
        """Click the 'Next' pagination button. Returns True if successful."""
        from selenium.webdriver.common.by import By
        from selenium.webdriver.support.ui import WebDriverWait
        from selenium.webdriver.support import expected_conditions as EC

        try:
            next_btn = self._driver.find_element(
                By.CSS_SELECTOR, "a.nav.next, a[data-page-number]"
            )
            if "disabled" in next_btn.get_attribute("class"):
                return False

            current_url = self._driver.current_url
            self._driver.execute_script("arguments[0].click();", next_btn)

            WebDriverWait(self._driver, 10).until(
                lambda d: d.current_url != current_url
            )
            self._sleep()
            return True

        except Exception:
            return False

    def collect(
        self,
        url: str,
        max_pages: int = 5,
        property_name: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Scrape reviews from a TripAdvisor property page.

        Args:
            url:           Full TripAdvisor URL of the hotel/restaurant
            max_pages:     Maximum number of review pages to scrape
            property_name: Optional label for the data

        Returns:
            DataFrame: text, title, rating, date, user_name, platform, property
        """
        self._init_driver()
        all_reviews = []

        try:
            print(f"🔍 TripAdvisor: {url[:60]}...")
            self._driver.get(url)
            self._sleep(extra=1)

            for page_num in range(1, max_pages + 1):
                self._expand_reviews()
                reviews = self._parse_page(url)
                all_reviews.extend(reviews)
                print(f"  Page {page_num}: {len(reviews)} reviews")

                if page_num < max_pages:
                    if not self._go_to_next_page():
                        print("  No more pages.")
                        break

        finally:
            self._close_driver()

        df = pd.DataFrame(all_reviews)
        if len(df) > 0:
            if property_name:
                df["property"] = property_name
            df = df.drop_duplicates(subset=["text"])
            df = df.reset_index(drop=True)

        print(f"✅ Total: {len(df)} TripAdvisor reviews")
        return df

    def collect_multiple(
        self,
        urls: List[str],
        max_pages: int = 3,
        names: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Scrape multiple TripAdvisor properties."""
        all_dfs = []

        for i, url in enumerate(urls):
            name = names[i] if names and i < len(names) else None
            df = self.collect(url, max_pages=max_pages, property_name=name)
            if len(df) > 0:
                all_dfs.append(df)

        if not all_dfs:
            return pd.DataFrame()

        combined = pd.concat(all_dfs, ignore_index=True)
        combined = combined.drop_duplicates(subset=["text"])
        print(f"\n✅ Grand total: {len(combined)} reviews")
        return combined
