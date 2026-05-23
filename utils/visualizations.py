"""
Plotly visualizations used across the Streamlit pages.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


SENTIMENT_COLORS = {
    "Positive": "#16a34a",
    "Neutral": "#64748b",
    "Negative": "#dc2626",
}

ASPECT_COLORS = {
    "Quality": "#2563eb",
    "Sizing": "#9333ea",
    "Delivery": "#0891b2",
    "Price": "#ca8a04",
    "Design": "#db2777",
}

PALETTE = ["#2563eb", "#16a34a", "#dc2626", "#ca8a04", "#9333ea", "#0891b2", "#db2777"]


class Visualizer:
    """Factory for reusable Plotly charts."""

    @staticmethod
    def empty_figure(title: str = "No data available") -> go.Figure:
        fig = go.Figure()
        fig.update_layout(
            title=title,
            height=320,
            template="plotly_white",
            xaxis={"visible": False},
            yaxis={"visible": False},
        )
        return fig

    @staticmethod
    def sentiment_pie(data, label_col: str = "vader_label") -> go.Figure:
        """Donut chart from either a DataFrame or a distribution dictionary."""
        if isinstance(data, pd.DataFrame):
            counts = data[label_col].value_counts() if label_col in data.columns else pd.Series()
            labels = ["Positive", "Neutral", "Negative"]
            values = [int(counts.get(label, 0)) for label in labels]
        else:
            labels = list(data.keys())
            values = [int(data[label]["count"]) for label in labels]

        fig = go.Figure(
            go.Pie(
                labels=labels,
                values=values,
                hole=0.55,
                marker_colors=[SENTIMENT_COLORS.get(label, "#64748b") for label in labels],
                textinfo="label+percent",
            )
        )
        fig.update_layout(title="Sentiment Distribution", height=360, template="plotly_white")
        return fig

    @staticmethod
    def sentiment_bar_by_category(
        df: pd.DataFrame,
        category_col: str = "category",
        label_col: str = "vader_label",
    ) -> go.Figure:
        if df.empty or label_col not in df.columns:
            return Visualizer.empty_figure("Sentiment by Category")

        data = df.copy()
        if category_col not in data.columns:
            data[category_col] = "All"

        grouped = data.groupby([category_col, label_col]).size().reset_index(name="count")
        fig = px.bar(
            grouped,
            x=category_col,
            y="count",
            color=label_col,
            barmode="group",
            color_discrete_map=SENTIMENT_COLORS,
            title="Sentiment by Category",
            labels={category_col: "Category", "count": "Reviews", label_col: "Sentiment"},
        )
        fig.update_layout(height=380, template="plotly_white")
        return fig

    @staticmethod
    def sentiment_bar(df: pd.DataFrame, label_col: str = "vader_label") -> go.Figure:
        return Visualizer.sentiment_bar_by_category(df, label_col=label_col)

    @staticmethod
    def platform_sentiment(df: pd.DataFrame) -> go.Figure:
        if df.empty or "platform" not in df.columns or "vader_label" not in df.columns:
            return Visualizer.empty_figure("Sentiment by Platform")
        grouped = df.groupby(["platform", "vader_label"]).size().reset_index(name="count")
        fig = px.bar(
            grouped,
            x="platform",
            y="count",
            color="vader_label",
            color_discrete_map=SENTIMENT_COLORS,
            barmode="stack",
            title="Sentiment by Platform",
        )
        fig.update_layout(height=340, template="plotly_white")
        return fig

    @staticmethod
    def rating_distribution(df: pd.DataFrame) -> go.Figure:
        if df.empty or "rating" not in df.columns:
            return Visualizer.empty_figure("Rating Distribution")
        counts = pd.to_numeric(df["rating"], errors="coerce").value_counts().sort_index()
        fig = px.bar(
            x=counts.index.astype(str),
            y=counts.values,
            color=counts.index,
            color_continuous_scale="RdYlGn",
            title="Rating Distribution",
            labels={"x": "Rating", "y": "Reviews"},
        )
        fig.update_layout(height=320, template="plotly_white", coloraxis_showscale=False)
        return fig

    @staticmethod
    def compound_histogram(df: pd.DataFrame, compound_col: str = "vader_compound") -> go.Figure:
        if df.empty or compound_col not in df.columns:
            return Visualizer.empty_figure("VADER Compound Scores")
        fig = px.histogram(
            df,
            x=compound_col,
            nbins=24,
            title="VADER Compound Score Distribution",
            color_discrete_sequence=["#2563eb"],
        )
        fig.add_vline(x=0.05, line_dash="dash", line_color=SENTIMENT_COLORS["Positive"])
        fig.add_vline(x=-0.05, line_dash="dash", line_color=SENTIMENT_COLORS["Negative"])
        fig.update_layout(height=320, template="plotly_white")
        return fig

    @staticmethod
    def monthly_sentiment_line(df: pd.DataFrame, label_col: str = "vader_label") -> go.Figure:
        if df.empty or "date" not in df.columns or label_col not in df.columns:
            return Visualizer.empty_figure("Monthly Sentiment Trend")
        data = df.copy()
        data["date"] = pd.to_datetime(data["date"], errors="coerce")
        data = data.dropna(subset=["date"])
        if data.empty:
            return Visualizer.empty_figure("Monthly Sentiment Trend")
        data["month"] = data["date"].dt.to_period("M").astype(str)
        grouped = data.groupby(["month", label_col]).size().reset_index(name="count")
        fig = px.line(
            grouped,
            x="month",
            y="count",
            color=label_col,
            markers=True,
            color_discrete_map=SENTIMENT_COLORS,
            title="Monthly Sentiment Trend",
            labels={"month": "Month", "count": "Reviews", label_col: "Sentiment"},
        )
        fig.update_layout(height=380, template="plotly_white")
        return fig

    @staticmethod
    def sentiment_over_time(df: pd.DataFrame, label_col: str = "vader_label") -> go.Figure:
        return Visualizer.monthly_sentiment_line(df, label_col=label_col)

    @staticmethod
    def aspect_sentiment_breakdown(aspect_df: pd.DataFrame) -> go.Figure:
        if aspect_df.empty:
            return Visualizer.empty_figure("Aspect Sentiment Breakdown")
        grouped = aspect_df.groupby(["aspect", "sentiment"]).size().reset_index(name="count")
        fig = px.bar(
            grouped,
            x="aspect",
            y="count",
            color="sentiment",
            barmode="stack",
            color_discrete_map=SENTIMENT_COLORS,
            title="Aspect-Based Sentiment Breakdown",
            labels={"aspect": "Aspect", "count": "Mentions"},
        )
        fig.update_layout(height=390, template="plotly_white")
        return fig

    @staticmethod
    def aspect_score_bars(summary_df: pd.DataFrame) -> go.Figure:
        if summary_df.empty:
            return Visualizer.empty_figure("Aspect Positive Rates")
        data = summary_df.sort_values("positive_rate", ascending=True)
        fig = px.bar(
            data,
            x="positive_rate",
            y="aspect",
            orientation="h",
            color="aspect",
            color_discrete_map=ASPECT_COLORS,
            title="Positive Mention Rate by Aspect",
            labels={"positive_rate": "Positive Mentions (%)", "aspect": "Aspect"},
            text="positive_rate",
        )
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig.update_layout(height=360, template="plotly_white", showlegend=False)
        fig.update_xaxes(range=[0, 105])
        return fig

    @staticmethod
    def aspect_trend_line(
        trend_df: pd.DataFrame,
        metric: str = "positive_rate",
        aspects: Optional[List[str]] = None,
    ) -> go.Figure:
        if trend_df.empty or metric not in trend_df.columns:
            return Visualizer.empty_figure("Aspect Trend")
        data = trend_df.copy()
        if aspects:
            data = data[data["aspect"].isin(aspects)]
        if data.empty:
            return Visualizer.empty_figure("Aspect Trend")

        fig = px.line(
            data,
            x="month",
            y=metric,
            color="aspect",
            markers=True,
            color_discrete_map=ASPECT_COLORS,
            title="Monthly Aspect Sentiment Trend",
            labels={"month": "Month", metric: metric.replace("_", " ").title()},
        )
        fig.update_layout(height=420, template="plotly_white")
        fig.update_yaxes(range=[0, 105])
        return fig

    @staticmethod
    def topic_distribution(df: pd.DataFrame) -> go.Figure:
        if df.empty or "topic_label" not in df.columns:
            return Visualizer.empty_figure("Topic Distribution")
        counts = df["topic_label"].value_counts().reset_index()
        counts.columns = ["topic_label", "count"]
        fig = px.bar(
            counts,
            x="count",
            y="topic_label",
            orientation="h",
            color="topic_label",
            color_discrete_sequence=PALETTE,
            title="Topic Distribution",
            labels={"count": "Reviews", "topic_label": "Topic"},
        )
        fig.update_layout(height=380, template="plotly_white", showlegend=False)
        return fig

    @staticmethod
    def topic_words_bar(topic: Dict) -> go.Figure:
        words = topic.get("words", [])[:10]
        weights = topic.get("weights", [])[:10]
        if not words:
            return Visualizer.empty_figure("Topic Keywords")
        fig = go.Figure(
            go.Bar(
                x=list(reversed(weights)),
                y=list(reversed(words)),
                orientation="h",
                marker_color="#2563eb",
            )
        )
        fig.update_layout(
            title=f"{topic.get('label', 'Topic')} Top Keywords",
            height=340,
            template="plotly_white",
            xaxis_title="Weight",
        )
        return fig

    @staticmethod
    def topic_trend_area(topic_trends: pd.DataFrame) -> go.Figure:
        if topic_trends.empty:
            return Visualizer.empty_figure("Topic Trends")
        fig = px.area(
            topic_trends,
            x="month",
            y="share",
            color="topic_label",
            title="Monthly Topic Share",
            labels={"month": "Month", "share": "Share of Reviews (%)", "topic_label": "Topic"},
        )
        fig.update_layout(height=420, template="plotly_white")
        return fig

    @staticmethod
    def decision_priority_bar(decisions: List[Dict]) -> go.Figure:
        if not decisions:
            return Visualizer.empty_figure("Decision Priorities")
        data = pd.DataFrame(decisions)
        counts = data["priority"].value_counts().reset_index()
        counts.columns = ["priority", "count"]
        color_map = {
            "🚨 Critical": "#dc2626",
            "⚠️ Warning": "#ca8a04",
            "✅ Good": "#16a34a",
        }
        fig = px.bar(
            counts,
            x="priority",
            y="count",
            color="priority",
            color_discrete_map=color_map,
            title="Recommendation Priority Mix",
            labels={"priority": "Priority", "count": "Recommendations"},
        )
        fig.update_layout(height=320, template="plotly_white", showlegend=False)
        return fig

    @staticmethod
    def bert_comparison(df: pd.DataFrame) -> go.Figure:
        if df.empty or "bert_label" not in df.columns or "vader_label" not in df.columns:
            return Visualizer.empty_figure("VADER vs DistilBERT")
        labels = ["Positive", "Neutral", "Negative"]
        vader = df["vader_label"].value_counts()
        bert = df["bert_label"].value_counts()
        fig = go.Figure()
        fig.add_bar(name="VADER", x=labels, y=[int(vader.get(label, 0)) for label in labels])
        fig.add_bar(name="DistilBERT", x=labels, y=[int(bert.get(label, 0)) for label in labels])
        fig.update_layout(
            title="VADER vs DistilBERT Sentiment Labels",
            barmode="group",
            height=340,
            template="plotly_white",
        )
        return fig
