from __future__ import annotations

import argparse
from pathlib import Path

from scraping.shein_scraper import SheinScraper


DEFAULT_URL = (
    "https://us.shein.com/Women-s-Polka-Dot-Print-Pocket-Wide-Leg-Pants-review-158214269.html"
    "?mall_code=1&goods_spu=z250814661199&local_site_query_flag=&sku=sz25081466119988888"
    "&store_code=2115186932&isLowestPriceProductOfBuyBox=NaN&mainProductSameGroupId=null"
    "&cat_id=1740&sku_code=&isAppointMall=&isTrusted=true&_vts=1778860038532"
)


def parse_args():
    parser = argparse.ArgumentParser(description="Scrape SHEIN product reviews.")
    parser.add_argument("--url", default=DEFAULT_URL, help="SHEIN product or review page URL.")
    parser.add_argument("--name", default="SHEIN Product", help="Optional product name saved in the CSV.")
    parser.add_argument("--pages", type=int, default=1, help="Maximum number of SHEIN review pages to scrape.")
    parser.add_argument("--headless", action="store_true", help="Run the browser headlessly.")
    parser.add_argument("--delay", type=float, default=3.0, help="Base delay between loads in seconds.")
    parser.add_argument(
        "--manual",
        action="store_true",
        help="Open the browser first and let you navigate or solve challenges manually before scraping.",
    )
    return parser.parse_args()


args = parse_args()
scraper = SheinScraper(headless=args.headless, delay=args.delay)

if args.manual:
    scraper.prepare_manual_session(start_url=args.url)
    df = scraper.collect_from_current_page(max_pages=args.pages, product_name=args.name)
else:
    df = scraper.collect(url=args.url, max_pages=args.pages, product_name=args.name)

Path("data").mkdir(exist_ok=True)
df.to_csv("data/shein_reviews.csv", index=False)

print(f"\nCollected {len(df)} SHEIN reviews")
if len(df) > 0:
    print(df[["text", "rating", "date"]].head(10))
else:
    print("No reviews were collected. SHEIN may still require a manual challenge solve or a different page layout.")