"""
Business KPI calculations for the clothing review dashboard.
"""

from __future__ import annotations

from typing import Dict

import pandas as pd

from models.sentiment_vader import sentiment_weight


def calculate_all_kpis(
    reviews_df: pd.DataFrame,
    aspect_mentions: pd.DataFrame,
    sentiment_col: str = "vader_label",
) -> Dict[str, Dict[str, float | int | str]]:
    """Calculate all business KPIs requested by the project brief."""
    total_reviews = max(len(reviews_df), 1)
    labels = reviews_df.get(sentiment_col, pd.Series(dtype=str)).fillna("Neutral")
    satisfaction = round(labels.map(sentiment_weight).mean() * 100, 1) if len(labels) else 0.0

    quality_score = _positive_mention_rate(aspect_mentions, "Quality")
    delivery_score = _positive_mention_rate(aspect_mentions, "Delivery")
    return_risk = _negative_review_rate(aspect_mentions, "Sizing", total_reviews)

    ratings = pd.to_numeric(reviews_df.get("rating", pd.Series(dtype=float)), errors="coerce")
    five_star = int((ratings == 5).sum())
    one_star = int((ratings == 1).sum())
    rated_total = int(ratings.notna().sum())
    ratio = round(five_star / one_star, 2) if one_star > 0 else float(five_star)
    net_promoter = round(((five_star - one_star) / max(rated_total, 1)) * 100, 1)

    sentiment_counts = labels.value_counts()

    return {
        "overall_customer_satisfaction": {
            "label": "Overall Customer Satisfaction",
            "value": satisfaction,
            "suffix": "%",
            "help": "Weighted sentiment average: Positive=100, Neutral=50, Negative=0.",
        },
        "quality_score": {
            "label": "Quality Score",
            "value": quality_score,
            "suffix": "%",
            "help": "Percent of quality mentions that are positive.",
        },
        "return_risk_index": {
            "label": "Return Risk Index",
            "value": return_risk,
            "suffix": "%",
            "help": "Percent of all reviews with negative sizing or fit mentions.",
        },
        "delivery_score": {
            "label": "Delivery Score",
            "value": delivery_score,
            "suffix": "%",
            "help": "Percent of delivery mentions that are positive.",
        },
        "net_promoter_sentiment": {
            "label": "Net Promoter Sentiment",
            "value": net_promoter,
            "suffix": "%",
            "ratio": ratio,
            "five_star": five_star,
            "one_star": one_star,
            "help": "Five-star to one-star ratio plus normalized promoter-minus-detractor score.",
        },
        "review_volume": {
            "label": "Review Volume",
            "value": int(len(reviews_df)),
            "suffix": "",
            "positive": int(sentiment_counts.get("Positive", 0)),
            "neutral": int(sentiment_counts.get("Neutral", 0)),
            "negative": int(sentiment_counts.get("Negative", 0)),
        },
    }


def _positive_mention_rate(aspect_mentions: pd.DataFrame, aspect: str) -> float:
    subset = _aspect_subset(aspect_mentions, aspect)
    if subset.empty:
        return 0.0
    return round((subset["sentiment"] == "Positive").mean() * 100, 1)


def _negative_review_rate(
    aspect_mentions: pd.DataFrame,
    aspect: str,
    total_reviews: int,
) -> float:
    subset = _aspect_subset(aspect_mentions, aspect)
    if subset.empty:
        return 0.0
    negative_reviews = subset[subset["sentiment"] == "Negative"]["review_id"].nunique()
    return round(negative_reviews / max(total_reviews, 1) * 100, 1)


def _aspect_subset(aspect_mentions: pd.DataFrame, aspect: str) -> pd.DataFrame:
    if aspect_mentions.empty or "aspect" not in aspect_mentions.columns:
        return pd.DataFrame()
    return aspect_mentions[aspect_mentions["aspect"] == aspect]
