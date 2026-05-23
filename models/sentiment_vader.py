"""
VADER sentiment analysis helpers.

The app uses VADER as the fast baseline classifier for every review and for
aspect-level text snippets.
"""

from __future__ import annotations

from typing import Dict

import pandas as pd

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
except ImportError:  # pragma: no cover - fallback for environments using NLTK only
    import nltk
    from nltk.sentiment.vader import SentimentIntensityAnalyzer

    try:
        nltk.data.find("sentiment/vader_lexicon.zip")
    except LookupError:
        nltk.download("vader_lexicon", quiet=True)


SENTIMENT_ORDER = ["Positive", "Neutral", "Negative"]


def label_from_compound(
    compound: float,
    pos_threshold: float = 0.05,
    neg_threshold: float = -0.05,
) -> str:
    """Convert a VADER compound score into a business-friendly label."""
    # Thresholds mirror common VADER defaults.
    if compound >= pos_threshold:
        return "Positive"
    if compound <= neg_threshold:
        return "Negative"
    return "Neutral"


def sentiment_weight(label: str) -> float:
    """Map a sentiment label to a KPI weight."""
    return {"Positive": 1.0, "Neutral": 0.5, "Negative": 0.0}.get(str(label), 0.5)


class VADERSentimentAnalyzer:
    """Rule-based sentiment classifier using VADER."""

    def __init__(self, pos_threshold: float = 0.05, neg_threshold: float = -0.05):
        self.analyzer = SentimentIntensityAnalyzer()
        self.pos_threshold = pos_threshold
        self.neg_threshold = neg_threshold

    def predict(self, text: str) -> Dict[str, float | str]:
        """Analyze a single text value."""
        scores = self.analyzer.polarity_scores(str(text))
        compound = float(scores["compound"])
        label = label_from_compound(
            compound,
            pos_threshold=self.pos_threshold,
            neg_threshold=self.neg_threshold,
        )

        confidence = {
            "Positive": scores["pos"],
            "Negative": scores["neg"],
            "Neutral": scores["neu"],
        }[label]

        return {
            "label": label,
            "compound": round(compound, 4),
            "positive": round(float(scores["pos"]), 4),
            "negative": round(float(scores["neg"]), 4),
            "neutral": round(float(scores["neu"]), 4),
            "confidence": round(float(confidence), 4),
        }

    def predict_batch(
        self,
        df: pd.DataFrame,
        text_col: str = "text",
        prefix: str = "vader_",
    ) -> pd.DataFrame:
        """Add VADER sentiment columns to a DataFrame."""
        if text_col not in df.columns:
            raise ValueError(f"Column '{text_col}' not found.")

        output = df.copy().reset_index(drop=True)
        results = output[text_col].fillna("").astype(str).apply(self.predict)
        results_df = pd.DataFrame(results.tolist()).add_prefix(prefix)
        return pd.concat([output, results_df], axis=1)

    def get_sentiment_distribution(
        self,
        df: pd.DataFrame,
        label_col: str = "vader_label",
    ) -> Dict[str, Dict[str, float | int]]:
        """Return count and percentage for each sentiment label."""
        if label_col not in df.columns:
            df = self.predict_batch(df)

        total = max(len(df), 1)
        counts = df[label_col].value_counts()
        return {
            label: {
                "count": int(counts.get(label, 0)),
                "percentage": round(float(counts.get(label, 0)) / total * 100, 1),
            }
            for label in SENTIMENT_ORDER
        }
