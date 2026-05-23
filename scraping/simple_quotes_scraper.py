"""
Simple beginner-friendly scraper.

This uses http://quotes.toscrape.com, a website made for practicing web scraping.
It collects quotes, authors, and tags, then saves them to data/simple_quotes.csv.
"""

from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup


BASE_URL = "http://quotes.toscrape.com"


def scrape_quotes(max_pages: int = 2) -> pd.DataFrame:
    quotes = []

    for page in range(1, max_pages + 1):
        url = f"{BASE_URL}/page/{page}/"
        print(f"Scraping page {page}: {url}")

        response = requests.get(url, timeout=15)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")
        quote_cards = soup.select(".quote")

        for card in quote_cards:
            text = card.select_one(".text").get_text(strip=True)
            author = card.select_one(".author").get_text(strip=True)
            tags = [tag.get_text(strip=True) for tag in card.select(".tag")]

            quotes.append(
                {
                    "text": text,
                    "author": author,
                    "tags": ", ".join(tags),
                    "source": "quotes.toscrape.com",
                }
            )

    return pd.DataFrame(quotes)


if __name__ == "__main__":
    output_path = Path(__file__).resolve().parents[1] / "data" / "simple_quotes.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = scrape_quotes(max_pages=2)
    df.to_csv(output_path, index=False)

    print(f"\nSaved {len(df)} quotes to {output_path}")
    print(df.head(5).to_string(index=False))
