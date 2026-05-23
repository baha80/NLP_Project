from pathlib import Path
import argparse

from scraping.amazon_scraper import AmazonScraper


DEFAULT_PRODUCT = (
    "https://www.amazon.com/Womens-Floral-Sleeve-Kimono-Cardigan/dp/B07SN9RS13"
)


def parse_args():
    parser = argparse.ArgumentParser(description="Scrape Amazon reviews for one product URL or ASIN.")
    parser.add_argument(
        "--product",
        default=DEFAULT_PRODUCT,
        help="Amazon product URL or ASIN to scrape.",
    )
    parser.add_argument(
        "--name",
        default="Women's Floral Sleeve Kimono Cardigan",
        help="Optional product name stored in the output CSV.",
    )
    parser.add_argument(
        "--pages",
        type=int,
        default=2,
        help="Maximum number of Amazon review pages to scrape.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run the browser headlessly. Amazon is often easier in visible mode.",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=4.0,
        help="Base delay between page loads in seconds.",
    )
    parser.add_argument(
        "--manual",
        action="store_true",
        help="Open Chrome first and let you navigate to the Amazon product page manually before scraping.",
    )
    return parser.parse_args()


args = parse_args()

scraper = AmazonScraper(headless=args.headless, delay=args.delay)

product_ref = args.product
if args.manual:
    product_ref = scraper.prepare_manual_session(start_url=args.product)

df = scraper.collect(
    product_url=product_ref,
    max_pages=args.pages,
    product_name=args.name,
)

Path("data").mkdir(exist_ok=True)
df.to_csv("data/amazon_reviews.csv", index=False)

print(f"\nCollected {len(df)} reviews")

if len(df) > 0:
    print(df[["text", "rating", "date"]].head(10))
else:
    print("No reviews were collected. Amazon may have blocked the request or shown a CAPTCHA.")
