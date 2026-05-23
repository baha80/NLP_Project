"""
Aspect-based sentiment analysis for clothing reviews.

Each configured aspect is detected with a domain keyword list. Sentiment is
scored on the sentence or local context where the keyword appears.
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Dict, Iterable, List, Tuple

import pandas as pd

from models.sentiment_vader import VADERSentimentAnalyzer


ASPECT_KEYWORDS: Dict[str, List[str]] = {
    "Quality": [
        "quality",
        "fabric",
        "material",
        "stitching",
        "construction",
        "seam",
        "seams",
        "thread",
        "threads",
        "button",
        "buttons",
        "zipper",
        "zippers",
        "durable",
        "durability",
        "cotton",
        "linen",
        "polyester",
        "wool",
        "pilling",
        "ripped",
        "tear",
        "hole",
        "holes",
        "shrunk",
        "shrink",
        "faded",
        "transparent",
        "see through",
    ],
    "Sizing": [
        "size",
        "sizes",
        "sizing",
        "fit",
        "fits",
        "fitted",
        "small",
        "large",
        "tight",
        "loose",
        "baggy",
        "oversized",
        "undersized",
        "waist",
        "length",
        "sleeve",
        "measurement",
        "measurements",
        "size chart",
        "true to size",
        "runs small",
        "runs large",
        "return",
        "exchange",
    ],
    "Delivery": [
        "delivery",
        "delivered",
        "shipping",
        "shipment",
        "arrived",
        "arrival",
        "fast",
        "slow",
        "delayed",
        "delay",
        "late",
        "courier",
        "logistics",
        "tracking",
        "dispatch",
        "packaging",
        "package",
    ],
    "Price": [
        "price",
        "pricing",
        "value",
        "expensive",
        "cheap",
        "affordable",
        "costly",
        "overpriced",
        "worth",
        "money",
        "discount",
        "bargain",
        "premium",
    ],
    "Design": [
        "design",
        "color",
        "colour",
        "pattern",
        "beautiful",
        "style",
        "stylish",
        "look",
        "looks",
        "print",
        "cut",
        "silhouette",
        "elegant",
        "modern",
        "classic",
        "fashion",
    ],
}

NEGATIVE_CUES: Dict[str, List[str]] = {
    "Quality": [
        "pilling",
        "see through",
        "thin",
        "cheap",
        "broke",
        "broken",
        "loose thread",
        "loose threads",
        "hole",
        "holes",
        "rough",
        "uneven",
        "shrunk",
        "faded",
        "defect",
        "defective",
    ],
    "Sizing": [
        "too tight",
        "too loose",
        "too small",
        "too large",
        "too long",
        "too short",
        "fit is loose",
        "bust area is tight",
        "waist is tight",
        "legs are loose",
        "sleeves are tight",
        "runs small",
        "runs large",
        "run small",
        "run large",
        "does not fit",
        "inconsistent",
        "had to exchange",
        "need to return",
        "returned",
        "return it",
        "wrong",
        "confusing",
        "unreliable",
        "shorter than",
        "longer than",
    ],
    "Delivery": [
        "delayed",
        "delay",
        "slow",
        "late",
        "took too long",
        "arrived after",
        "crushed",
        "torn",
        "poor packaging",
        "tracking stopped",
        "tracking did not update",
    ],
    "Price": [
        "overpriced",
        "expensive",
        "costly",
        "not worth",
        "expected better",
        "premium price but",
    ],
    "Design": ["dull", "not like the photos", "different from the photos"],
}

POSITIVE_CUES: Dict[str, List[str]] = {
    "Quality": [
        "premium",
        "durable",
        "neat stitching",
        "clean construction",
        "soft",
        "breathable",
        "comfortable",
        "strong stitching",
    ],
    "Sizing": [
        "true to size",
        "fits true to size",
        "fits well",
        "fit perfectly",
        "fits perfectly",
        "perfect fit",
    ],
    "Delivery": ["fast", "early", "on time", "quickly", "smooth", "careful packaging"],
    "Price": ["great value", "worth", "affordable", "bargain"],
    "Design": ["beautiful", "elegant", "stylish", "modern", "classic", "gorgeous"],
}


@lru_cache(maxsize=1)
def _load_sentence_model():
    """Load spaCy for sentence segmentation, falling back to a blank model."""
    try:
        import spacy

        try:
            nlp = spacy.load("en_core_web_sm", disable=["ner"])
        except OSError:
            nlp = spacy.blank("en")
        if "sentencizer" not in nlp.pipe_names:
            nlp.add_pipe("sentencizer")
        return nlp
    except Exception:
        return None


def _compile_keyword_pattern(keyword: str) -> re.Pattern:
    escaped = re.escape(keyword.lower()).replace(r"\ ", r"\s+")
    return re.compile(rf"(?<!\w){escaped}(?!\w)", flags=re.IGNORECASE)


class AspectSentimentAnalyzer:
    """Detect clothing review aspects and score the sentiment around them."""

    def __init__(self, aspects: Dict[str, List[str]] | None = None):
        self.aspects = aspects or ASPECT_KEYWORDS
        self.sentiment = VADERSentimentAnalyzer()
        # Pre-compile keyword patterns for fast aspect matching.
        self._patterns = {
            aspect: [(keyword, _compile_keyword_pattern(keyword)) for keyword in keywords]
            for aspect, keywords in self.aspects.items()
        }

    def analyze(
        self,
        df: pd.DataFrame,
        text_col: str = "text",
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Return an enriched review table and a long-form aspect mention table."""
        if text_col not in df.columns:
            raise ValueError(f"Column '{text_col}' not found.")

        output = df.copy().reset_index(drop=True)
        mentions: list[dict] = []

        for aspect in self.aspects:
            slug = self._slug(aspect)
            output[f"{slug}_mentioned"] = False
            output[f"{slug}_sentiment"] = "Not Mentioned"
            output[f"{slug}_compound"] = 0.0
            output[f"{slug}_keywords"] = ""
            output[f"{slug}_context"] = ""

        for idx, row in output.iterrows():
            text = str(row.get(text_col, ""))
            sentences = self._sentences(text)
            review_id = row.get("review_id", idx + 1)

            for aspect in self.aspects:
                matched = self._matched_keywords(text, aspect)
                if not matched:
                    continue

                contexts = self._contexts_for_keywords(sentences, text, aspect)
                context_text = " ".join(contexts) if contexts else text
                result = self.sentiment.predict(context_text)
                result = self._apply_cue_overrides(aspect, context_text, result)
                slug = self._slug(aspect)

                output.at[idx, f"{slug}_mentioned"] = True
                output.at[idx, f"{slug}_sentiment"] = result["label"]
                output.at[idx, f"{slug}_compound"] = result["compound"]
                output.at[idx, f"{slug}_keywords"] = ", ".join(matched)
                output.at[idx, f"{slug}_context"] = context_text

                mentions.append(
                    {
                        "review_id": review_id,
                        "text": text,
                        "rating": row.get("rating"),
                        "date": row.get("date"),
                        "category": row.get("category", "Unknown"),
                        "platform": row.get("platform", "Unknown"),
                        "aspect": aspect,
                        "sentiment": result["label"],
                        "compound": result["compound"],
                        "keywords": ", ".join(matched),
                        "context": context_text,
                    }
                )

        return output, pd.DataFrame(mentions)

    def aspect_summary(self, aspect_mentions: pd.DataFrame) -> pd.DataFrame:
        """Aggregate aspect mentions by sentiment."""
        if aspect_mentions.empty:
            return pd.DataFrame(
                columns=[
                    "aspect",
                    "mentions",
                    "positive",
                    "neutral",
                    "negative",
                    "positive_rate",
                    "negative_rate",
                ]
            )

        pivot = (
            aspect_mentions.pivot_table(
                index="aspect",
                columns="sentiment",
                values="review_id",
                aggfunc="count",
                fill_value=0,
            )
            .reset_index()
            .rename_axis(None, axis=1)
        )

        for label in ["Positive", "Neutral", "Negative"]:
            if label not in pivot.columns:
                pivot[label] = 0

        pivot["mentions"] = pivot[["Positive", "Neutral", "Negative"]].sum(axis=1)
        pivot["positive_rate"] = (pivot["Positive"] / pivot["mentions"] * 100).round(1)
        pivot["negative_rate"] = (pivot["Negative"] / pivot["mentions"] * 100).round(1)
        return pivot.rename(
            columns={
                "Positive": "positive",
                "Neutral": "neutral",
                "Negative": "negative",
            }
        ).sort_values("mentions", ascending=False)

    def trend_by_month(self, aspect_mentions: pd.DataFrame) -> pd.DataFrame:
        """Calculate monthly positive and negative rates per aspect."""
        if aspect_mentions.empty or "date" not in aspect_mentions.columns:
            return pd.DataFrame()

        data = aspect_mentions.copy()
        data["date"] = pd.to_datetime(data["date"], errors="coerce")
        data = data.dropna(subset=["date"])
        if data.empty:
            return pd.DataFrame()

        data["month"] = data["date"].dt.to_period("M").astype(str)
        grouped = (
            data.groupby(["month", "aspect", "sentiment"])
            .size()
            .reset_index(name="count")
        )
        pivot = (
            grouped.pivot_table(
                index=["month", "aspect"],
                columns="sentiment",
                values="count",
                fill_value=0,
            )
            .reset_index()
            .rename_axis(None, axis=1)
        )

        for label in ["Positive", "Neutral", "Negative"]:
            if label not in pivot.columns:
                pivot[label] = 0

        pivot["mentions"] = pivot[["Positive", "Neutral", "Negative"]].sum(axis=1)
        pivot["positive_rate"] = (pivot["Positive"] / pivot["mentions"] * 100).round(1)
        pivot["negative_rate"] = (pivot["Negative"] / pivot["mentions"] * 100).round(1)
        pivot["neutral_rate"] = (pivot["Neutral"] / pivot["mentions"] * 100).round(1)
        return pivot.sort_values(["aspect", "month"]).reset_index(drop=True)

    def trend_directions(self, trend_df: pd.DataFrame) -> pd.DataFrame:
        """Label each aspect as improving, declining, stable, or sparse."""
        if trend_df.empty:
            return pd.DataFrame(columns=["aspect", "direction", "delta_points"])

        rows = []
        for aspect, group in trend_df.groupby("aspect"):
            group = group.sort_values("month")
            if len(group) < 2:
                rows.append(
                    {"aspect": aspect, "direction": "Not enough data", "delta_points": 0.0}
                )
                continue

            first = float(group.iloc[0]["positive_rate"])
            last = float(group.iloc[-1]["positive_rate"])
            delta = round(last - first, 1)
            if delta >= 5:
                direction = "Improving"
            elif delta <= -5:
                direction = "Declining"
            else:
                direction = "Stable"
            rows.append({"aspect": aspect, "direction": direction, "delta_points": delta})

        return pd.DataFrame(rows).sort_values("aspect")

    def _matched_keywords(self, text: str, aspect: str) -> List[str]:
        lower_text = str(text).lower()
        matched = [
            keyword
            for keyword, pattern in self._patterns[aspect]
            if pattern.search(lower_text)
        ]
        return sorted(set(matched), key=matched.index)

    def _contexts_for_keywords(
        self,
        sentences: List[str],
        fallback_text: str,
        aspect: str,
    ) -> List[str]:
        contexts = []
        for sentence in sentences:
            if self._matched_keywords(sentence, aspect):
                contexts.append(sentence.strip())
        return contexts[:3] if contexts else [fallback_text]

    def _apply_cue_overrides(self, aspect: str, text: str, result: dict) -> dict:
        """Correct VADER for deterministic operational cues."""
        lower_text = str(text).lower()
        has_negative = any(cue in lower_text for cue in NEGATIVE_CUES.get(aspect, []))
        has_positive = any(cue in lower_text for cue in POSITIVE_CUES.get(aspect, []))
        updated = dict(result)

        if has_negative and not has_positive:
            updated["label"] = "Negative"
            updated["compound"] = min(float(updated.get("compound", 0.0)), -0.34)
            updated["confidence"] = max(float(updated.get("confidence", 0.0)), 0.7)
        elif has_positive and not has_negative:
            updated["label"] = "Positive"
            updated["compound"] = max(float(updated.get("compound", 0.0)), 0.34)
            updated["confidence"] = max(float(updated.get("confidence", 0.0)), 0.7)
        return updated

    @staticmethod
    def _sentences(text: str) -> List[str]:
        clean_text = str(text).strip()
        if not clean_text:
            return []

        nlp = _load_sentence_model()
        if nlp is not None:
            doc = nlp(clean_text)
            sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
            if sentences:
                return sentences

        return [s.strip() for s in re.split(r"(?<=[.!?])\s+", clean_text) if s.strip()]

    @staticmethod
    def _slug(aspect: str) -> str:
        return aspect.lower().replace(" ", "_")
