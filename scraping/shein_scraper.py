"""
shein_scraper.py
----------------
Scrape product reviews from SHEIN using Selenium with an optional manual
browser-first flow.

This scraper is designed for challenge-heavy pages where a normal requests
fetch is blocked. Open a visible browser, solve any risk challenge manually,
then let the scraper extract reviews from the loaded DOM.
"""

from __future__ import annotations

import os
import random
import re
import time
from datetime import datetime
from typing import List, Optional

import pandas as pd
from bs4 import BeautifulSoup, Tag


class SheinScraper:
    REVIEW_CARD_SELECTORS = [
        "div[class*='rate-comment-item']",
        "div[class*='rateCommentItem']",
        "div[class*='comment-item']",
        "li[class*='comment-item']",
        "div[class*='review-item']",
        "li[class*='review-item']",
        "article[class*='comment']",
        "article[class*='review']",
        "div[class*='product-review']",
        "div[data-testid*='comment']",
        "div[data-testid*='review']",
    ]

    NEXT_PAGE_SELECTORS = [
        "button[aria-label='Next']",
        "a[aria-label='Next']",
        "button[class*='next']",
        "a[class*='next']",
        "li[class*='next'] button",
        "li[class*='next'] a",
    ]

    MORE_BUTTON_XPATH = (
        "//*[self::button or self::span or self::a][contains(translate(normalize-space(.),"
        "'MORESEELESS', 'moreseeless'), 'more') or contains(translate(@aria-label,"
        "'MORESEELESS', 'moreseeless'), 'more')]"
    )

    DATE_PATTERNS = [
        "%b %d, %Y",
        "%B %d, %Y",
        "%Y-%m-%d",
        "%m/%d/%Y",
        "%d/%m/%Y",
    ]

    def __init__(self, headless: bool = False, delay: float = 3.0):
        self.headless = headless
        self.delay = delay
        self._driver = None

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
            self._driver = uc.Chrome(options=options)
            self._disable_driver_destructor(self._driver)
            print("✅ Undetected Chrome initialized")

        except ImportError:
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
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36"
            )
            service = Service(ChromeDriverManager().install())
            self._driver = webdriver.Chrome(service=service, options=options)
            self._driver.execute_script(
                "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"
            )
            print("✅ Chrome WebDriver initialized")

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

    def _sleep(self, extra: float = 0.0):
        time.sleep(self.delay + random.uniform(0.8, 2.2) + extra)

    def _is_challenge(self) -> bool:
        if self._driver is None:
            return False

        current_url = self._driver.current_url.lower()
        source = self._driver.page_source.lower()
        title = self._driver.title.lower()
        return (
            "/risk/challenge" in current_url
            or "captcha" in source
            or "risk challenge" in source
            or "security check" in title
            or "verify" in title
        )

    def _handle_challenge(self):
        print("\n" + "=" * 60)
        print("🛑 SHEIN CHALLENGE DETECTED")
        print("   A browser window is open.")
        print("   Solve the challenge or sign-in prompt there.")
        print("   Then return here and press ENTER to continue.")
        print("=" * 60)
        input("   >>> Press ENTER after the page is usable: ")
        self._sleep(1.5)

    def prepare_manual_session(self, start_url: Optional[str] = None) -> str:
        self._init_driver()
        target_url = start_url or "https://us.shein.com/"
        self._driver.get(target_url)

        while True:
            print("\n" + "=" * 60)
            print("🧭 MANUAL SHEIN SESSION")
            print("   Use the open Chrome window to:")
            print("   1. Solve any challenge or login prompt")
            print("   2. Open the SHEIN product review page you want to scrape")
            print("   3. Wait until the reviews are visible")
            print("=" * 60)
            input("   >>> Press ENTER when the review page is ready: ")

            current_url = self._driver.current_url
            if self._looks_like_shein_product_url(current_url):
                print(f"   Using current page: {current_url}")
                return current_url

            print(f"   Current page is not a SHEIN product/review page: {current_url}")
            print("   Open the review page, then press ENTER again.\n")

    @staticmethod
    def _looks_like_shein_product_url(url: str) -> bool:
        lower = str(url).lower()
        return "shein.com" in lower and (
            "review-" in lower or "goods_spu=" in lower or lower.endswith(".html")
        )

    def _scroll_reviews(self):
        if self._driver is None:
            return
        for _ in range(4):
            self._driver.execute_script("window.scrollBy(0, Math.floor(window.innerHeight * 0.85));")
            self._sleep(0.3)

    def _expand_review_texts(self):
        if self._driver is None:
            return

        from selenium.webdriver.common.by import By

        try:
            buttons = self._driver.find_elements(By.XPATH, self.MORE_BUTTON_XPATH)
        except Exception:
            buttons = []

        for button in buttons[:12]:
            try:
                self._driver.execute_script("arguments[0].click();", button)
                time.sleep(0.2)
            except Exception:
                continue

    def _page_soup(self) -> BeautifulSoup:
        return BeautifulSoup(self._driver.page_source, "html.parser")

    def _find_review_cards(self, soup: BeautifulSoup) -> List[Tag]:
        cards: list[Tag] = []
        for selector in self.REVIEW_CARD_SELECTORS:
            matches = soup.select(selector)
            if matches:
                cards.extend(matches)
        if cards:
            return cards

        fallback = soup.find_all(
            lambda tag: tag.name in {"div", "li", "article"}
            and tag.get("class")
            and re.search(r"comment|review", " ".join(tag.get("class", [])), re.IGNORECASE)
        )
        return fallback

    def _extract_review_text(self, card: Tag) -> str:
        candidates: list[tuple[int, str]] = []
        for node in card.find_all(["p", "div", "span", "li"]):
            text = re.sub(r"\s+", " ", " ".join(node.stripped_strings)).strip()
            if not 20 <= len(text) <= 1200:
                continue
            class_blob = " ".join(node.get("class", []))
            score = 0
            if re.search(r"comment|content|desc|text|review", class_blob, re.IGNORECASE):
                score += 3
            score += min(len(text) // 80, 5)
            candidates.append((score, text))

        if not candidates:
            text = re.sub(r"\s+", " ", card.get_text(" ", strip=True)).strip()
            return text if len(text) >= 20 else ""

        seen = set()
        deduped = []
        for score, text in sorted(candidates, key=lambda item: (item[0], len(item[1])), reverse=True):
            if text in seen:
                continue
            seen.add(text)
            deduped.append((score, text))

        return deduped[0][1] if deduped else ""

    def _extract_user(self, card: Tag) -> str:
        for node in card.find_all(["span", "div", "p"]):
            class_blob = " ".join(node.get("class", []))
            if not re.search(r"user|name|nick", class_blob, re.IGNORECASE):
                continue
            text = re.sub(r"\s+", " ", node.get_text(" ", strip=True)).strip()
            if 1 < len(text) <= 80:
                return text
        return "Anonymous"

    def _extract_rating(self, card: Tag) -> Optional[float]:
        for attr in ["aria-label", "title", "alt"]:
            for node in card.find_all(attrs={attr: True}):
                value = str(node.get(attr, ""))
                if "star" not in value.lower() and "rating" not in value.lower():
                    continue
                match = re.search(r"(\d(?:\.\d)?)", value)
                if match:
                    try:
                        return float(match.group(1))
                    except ValueError:
                        pass

        for node in card.find_all(["span", "div"]):
            class_blob = " ".join(node.get("class", []))
            if not re.search(r"rate|rating|score|star", class_blob, re.IGNORECASE):
                continue
            text = node.get_text(" ", strip=True)
            match = re.search(r"\b([1-5](?:\.\d)?)\b", text)
            if match:
                try:
                    return float(match.group(1))
                except ValueError:
                    pass
        return None

    def _extract_date(self, card: Tag) -> str:
        if time_node := card.find("time"):
            raw = time_node.get("datetime") or time_node.get_text(" ", strip=True)
            parsed = self._parse_date(raw)
            if parsed:
                return parsed

        for node in card.find_all(["span", "div", "p"]):
            text = re.sub(r"\s+", " ", node.get_text(" ", strip=True)).strip()
            parsed = self._parse_date(text)
            if parsed:
                return parsed

        return ""

    def _parse_date(self, raw: str) -> str:
        raw = str(raw).strip()
        if not raw:
            return ""

        raw = raw.replace("Reviewed on", "").strip()
        for fmt in self.DATE_PATTERNS:
            try:
                return datetime.strptime(raw, fmt).strftime("%Y-%m-%d")
            except ValueError:
                continue

        for pattern in [
            r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},\s+\d{4}\b",
            r"\b\d{4}-\d{2}-\d{2}\b",
            r"\b\d{1,2}/\d{1,2}/\d{4}\b",
        ]:
            match = re.search(pattern, raw, flags=re.IGNORECASE)
            if match:
                return self._parse_date(match.group(0))
        return ""

    def _extract_product_name(self, soup: BeautifulSoup) -> str:
        for selector in ["h1", "title", "meta[property='og:title']"]:
            node = soup.select_one(selector)
            if not node:
                continue
            if node.name == "meta":
                text = node.get("content", "")
            else:
                text = node.get_text(" ", strip=True)
            text = re.sub(r"\s+", " ", text).strip()
            if text:
                return text
        return ""

    def _extract_identifiers(self, url: str) -> dict:
        identifiers = {}
        for key in ["goods_spu", "sku", "cat_id"]:
            match = re.search(rf"[?&]{key}=([^&]+)", url)
            if match:
                identifiers[key] = match.group(1)
        return identifiers

    def _parse_current_page(self, product_name: Optional[str] = None) -> List[dict]:
        soup = self._page_soup()
        cards = self._find_review_cards(soup)
        current_url = self._driver.current_url
        identifiers = self._extract_identifiers(current_url)
        resolved_name = product_name or self._extract_product_name(soup) or "SHEIN Product"

        reviews = []
        for card in cards:
            text = self._extract_review_text(card)
            if len(text) < 20:
                continue

            reviews.append(
                {
                    "text": text,
                    "rating": self._extract_rating(card),
                    "date": self._extract_date(card),
                    "user_name": self._extract_user(card),
                    "product": resolved_name,
                    "platform": "SHEIN",
                    "url": current_url,
                    **identifiers,
                }
            )

        return reviews

    def _go_to_next_page(self, page_number: int) -> bool:
        if self._driver is None:
            return False

        from selenium.webdriver.common.by import By

        for selector in self.NEXT_PAGE_SELECTORS:
            try:
                elements = self._driver.find_elements(By.CSS_SELECTOR, selector)
            except Exception:
                elements = []

            for element in elements:
                try:
                    if not element.is_displayed() or not element.is_enabled():
                        continue
                    self._driver.execute_script("arguments[0].click();", element)
                    self._sleep(1.0)
                    return True
                except Exception:
                    continue

        return False

    def _collect_loaded_session(
        self,
        max_pages: int = 1,
        product_name: Optional[str] = None,
    ) -> pd.DataFrame:
        all_reviews = []
        for page in range(1, max_pages + 1):
            if self._is_challenge():
                self._handle_challenge()

            self._scroll_reviews()
            self._expand_review_texts()
            page_reviews = self._parse_current_page(product_name=product_name)

            if not page_reviews:
                print(f"  Page {page}: no reviews found — stopping")
                break

            all_reviews.extend(page_reviews)
            print(f"  Page {page}: {len(page_reviews)} reviews (total: {len(all_reviews)})")

            if page >= max_pages:
                break
            if not self._go_to_next_page(page):
                print("  No next page button found — stopping")
                break

        df = pd.DataFrame(all_reviews)
        if not df.empty:
            df = df.drop_duplicates(subset=["text", "url"]).reset_index(drop=True)
        print(f"✅ Total: {len(df)} reviews collected")
        return df

    def collect(
        self,
        url: str,
        max_pages: int = 1,
        product_name: Optional[str] = None,
    ) -> pd.DataFrame:
        self._init_driver()
        try:
            self._driver.get(url)
            self._sleep(1.5)
            return self._collect_loaded_session(max_pages=max_pages, product_name=product_name)
        finally:
            self._close_driver()

    def collect_from_current_page(
        self,
        max_pages: int = 1,
        product_name: Optional[str] = None,
    ) -> pd.DataFrame:
        if self._driver is None:
            raise RuntimeError("Browser is not initialized. Call prepare_manual_session() first.")
        try:
            return self._collect_loaded_session(max_pages=max_pages, product_name=product_name)
        finally:
            self._close_driver()


if __name__ == "__main__":
    scraper = SheinScraper(headless=False, delay=3.0)
    url = "https://us.shein.com/"
    scraper.prepare_manual_session(start_url=url)
    df = scraper.collect_from_current_page(max_pages=1)

    if len(df) > 0:
        output_path = os.path.join(os.path.dirname(__file__), "..", "data", "shein_reviews.csv")
        output_path = os.path.normpath(output_path)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"\n✅ Saved {len(df)} reviews → {output_path}")
    else:
        print("\n⚠️  0 reviews collected.")