"""
Automatic business recommendation engine.
"""

from __future__ import annotations

from typing import Dict, List

import pandas as pd

class DecisionEngine:
    """Turn sentiment, aspect and topic metrics into factory actions."""

    PRIORITY_ORDER = {"🚨 Critical": 0, "⚠️ Warning": 1, "✅ Good": 2}

    def generate(
        self,
        reviews_df: pd.DataFrame,
        aspect_mentions: pd.DataFrame,
        topic_trends: pd.DataFrame | None = None,
    ) -> List[Dict[str, str | float]]:
        total_reviews = max(len(reviews_df), 1)
        decisions: list[dict] = []

        # Threshold-based rules translate sentiment rates into business actions.
        sizing_complaints = self._negative_review_rate(
            aspect_mentions, "Sizing", total_reviews
        )
        quality_negative = self._negative_mention_rate(aspect_mentions, "Quality")
        delivery_negative = self._negative_mention_rate(aspect_mentions, "Delivery")
        price_negative = self._negative_mention_rate(aspect_mentions, "Price")

        if sizing_complaints > 35:
            self._add(
                decisions,
                "🚨 Critical",
                "Sizing returns are high",
                "Revise your size chart and compare garment measurements against the published guide.",
                sizing_complaints,
                25,
                "Sizing",
            )
        elif sizing_complaints > 25:
            self._add(
                decisions,
                "⚠️ Warning",
                "Sizing complaints exceed threshold",
                "Revise your size chart and add clearer fit notes on product pages.",
                sizing_complaints,
                25,
                "Sizing",
            )
        else:
            self._add(
                decisions,
                "✅ Good",
                "Sizing risk is under control",
                "Keep monitoring fit feedback by category before the next production batch.",
                sizing_complaints,
                25,
                "Sizing",
            )

        if quality_negative > 30:
            self._add(
                decisions,
                "🚨 Critical",
                "Quality complaints are elevated",
                "Investigate production quality, especially fabric, stitching and finishing checks.",
                quality_negative,
                20,
                "Quality",
            )
        elif quality_negative > 20:
            self._add(
                decisions,
                "⚠️ Warning",
                "Quality negative sentiment is above threshold",
                "Audit the current batch and inspect recurring fabric or stitching defects.",
                quality_negative,
                20,
                "Quality",
            )
        else:
            self._add(
                decisions,
                "✅ Good",
                "Quality sentiment is healthy",
                "Keep the existing quality control cadence and reuse praised materials.",
                quality_negative,
                20,
                "Quality",
            )

        if delivery_negative > 25:
            self._add(
                decisions,
                "🚨 Critical",
                "Delivery issues are damaging reviews",
                "Review your logistics partner and tighten dispatch, tracking and packaging controls.",
                delivery_negative,
                15,
                "Delivery",
            )
        elif delivery_negative > 15:
            self._add(
                decisions,
                "⚠️ Warning",
                "Delivery negative sentiment is above threshold",
                "Review your logistics partner and communicate delays earlier to customers.",
                delivery_negative,
                15,
                "Delivery",
            )
        else:
            self._add(
                decisions,
                "✅ Good",
                "Delivery sentiment is acceptable",
                "Maintain current shipping SLAs and keep monitoring late-arrival mentions.",
                delivery_negative,
                15,
                "Delivery",
            )

        if price_negative > 25:
            self._add(
                decisions,
                "⚠️ Warning",
                "Price-value complaints need attention",
                "Review pricing, discounts and product-page value messaging for low-rated categories.",
                price_negative,
                25,
                "Price",
            )

        for spike in self.detect_topic_spikes(topic_trends):
            self._add(
                decisions,
                "⚠️ Warning",
                "New recurring complaint detected",
                f"Alert: '{spike['topic_label']}' spiked this month vs last month. Inspect recent reviews before it grows.",
                float(spike["share_delta"]),
                15,
                "Topic Modeling",
            )

        return sorted(
            decisions,
            key=lambda item: (self.PRIORITY_ORDER.get(str(item["priority"]), 99), item["category"]),
        )

    def detect_topic_spikes(self, topic_trends: pd.DataFrame | None) -> List[Dict]:
        """Detect latest-month topic increases versus the previous month."""
        if topic_trends is None or topic_trends.empty:
            return []
        if not {"month", "topic_label", "count", "share"}.issubset(topic_trends.columns):
            return []

        data = topic_trends.copy().sort_values("month")
        months = sorted(data["month"].dropna().unique())
        if len(months) < 2:
            return []

        previous_month, current_month = months[-2], months[-1]
        previous = data[data["month"] == previous_month].set_index("topic_label")
        current = data[data["month"] == current_month].set_index("topic_label")
        spikes = []

        for topic_label, row in current.iterrows():
            current_count = int(row["count"])
            current_share = float(row["share"])
            previous_count = int(previous["count"].get(topic_label, 0))
            previous_share = float(previous["share"].get(topic_label, 0.0))
            count_delta = current_count - previous_count
            share_delta = round(current_share - previous_share, 1)

            if current_count >= 3 and share_delta >= 15:
                spikes.append(
                    {
                        "topic_label": topic_label,
                        "previous_month": previous_month,
                        "current_month": current_month,
                        "count_delta": count_delta,
                        "share_delta": share_delta,
                    }
                )

        return spikes

    @staticmethod
    def _negative_review_rate(
        aspect_mentions: pd.DataFrame,
        aspect: str,
        total_reviews: int,
    ) -> float:
        if aspect_mentions.empty:
            return 0.0
        subset = aspect_mentions[
            (aspect_mentions["aspect"] == aspect)
            & (aspect_mentions["sentiment"] == "Negative")
        ]
        return round(subset["review_id"].nunique() / total_reviews * 100, 1)

    @staticmethod
    def _negative_mention_rate(aspect_mentions: pd.DataFrame, aspect: str) -> float:
        if aspect_mentions.empty:
            return 0.0
        subset = aspect_mentions[aspect_mentions["aspect"] == aspect]
        if subset.empty:
            return 0.0
        return round((subset["sentiment"] == "Negative").mean() * 100, 1)

    @staticmethod
    def _add(
        decisions: list,
        priority: str,
        title: str,
        recommendation: str,
        value: float,
        threshold: float,
        category: str,
    ) -> None:
        decisions.append(
            {
                "priority": priority,
                "category": category,
                "title": title,
                "recommendation": recommendation,
                "metric_value": round(float(value), 1),
                "threshold": round(float(threshold), 1),
            }
        )
