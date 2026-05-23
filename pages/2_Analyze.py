from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.ner import NamedEntityRecognizer
from models.sentiment_bert import BERTSentimentAnalyzer
from models.sentiment_supervised import SupervisedSentimentAnalyzer
from models.sentiment_vader import VADERSentimentAnalyzer
from utils.visualizations import Visualizer


st.set_page_config(page_title="Real-Time Analysis", page_icon="🔍", layout="wide")


def load_css() -> None:
    css_path = ROOT / "assets" / "style.css"
    if css_path.exists():
        st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


@st.cache_resource
def get_supervised_model() -> SupervisedSentimentAnalyzer:
    return SupervisedSentimentAnalyzer(
        data_path=ROOT / "data" / "reviews.csv",
        fallback_path=ROOT / "data" / "sample_reviews.csv",
    )


def sentiment_gauge(compound: float) -> go.Figure:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=compound,
            number={"format": ".2f"},
            gauge={
                "axis": {"range": [-1, 1]},
                "bar": {"color": "#2563eb"},
                "steps": [
                    {"range": [-1, -0.05], "color": "#fee2e2"},
                    {"range": [-0.05, 0.05], "color": "#e2e8f0"},
                    {"range": [0.05, 1], "color": "#dcfce7"},
                ],
                "threshold": {"line": {"color": "#0f172a", "width": 3}, "value": compound},
            },
            title={"text": "VADER Compound"},
        )
    )
    fig.update_layout(height=300, template="plotly_white")
    return fig


load_css()

vader = VADERSentimentAnalyzer()
supervised = get_supervised_model()
ner = NamedEntityRecognizer()

st.title("Real-Time Review Analysis")
st.caption("Analyze one review, a small batch, or compare VADER, supervised ML, and DistilBERT on demand.")

tab_single, tab_batch, tab_compare = st.tabs(["Single Review", "Batch Reviews", "Model Comparison"])

with tab_single:
    st.subheader("Analyze a Single Review")
    model_choice = st.radio(
        "Model",
        ["VADER (fast)", "Supervised ML (TF-IDF + Logistic Regression)"],
        horizontal=True,
    )
    sample_texts = [
        "Select an example or type your own...",
        "The fabric feels premium, the fit is true to size, and delivery arrived early.",
        "Beautiful color, but the zipper broke and the size chart was misleading.",
        "Shipping was slow and the packaging was damaged, although the dress design is elegant.",
        "Affordable and stylish, but the stitching around the hem is uneven.",
    ]
    selected = st.selectbox("Quick examples", sample_texts)
    default_text = "" if selected == sample_texts[0] else selected
    user_text = st.text_area(
        "Review text",
        value=default_text,
        height=140,
        placeholder="Type or paste a customer review here...",
    )

    if st.button("Analyze Review", type="primary", key="single_review_btn"):
        if not user_text.strip():
            st.warning("Enter a review to analyze.")
        else:
            entities = ner.extract(user_text)
            entity_df = pd.DataFrame(entities)

            if model_choice.startswith("VADER"):
                result = vader.predict(user_text)

                cols = st.columns(4)
                cols[0].metric("Sentiment", result["label"])
                cols[1].metric("Compound", f"{result['compound']:.4f}")
                cols[2].metric("Confidence", f"{result['confidence']:.1%}")
                cols[3].metric("Entities", len(entities))

                left, right = st.columns(2)
                with left:
                    st.plotly_chart(sentiment_gauge(result["compound"]), use_container_width=True)
                with right:
                    breakdown = go.Figure(
                        go.Bar(
                            x=["Positive", "Neutral", "Negative"],
                            y=[result["positive"], result["neutral"], result["negative"]],
                            marker_color=["#16a34a", "#64748b", "#dc2626"],
                        )
                    )
                    breakdown.update_layout(
                        title="Score Breakdown",
                        height=300,
                        template="plotly_white",
                        yaxis_range=[0, 1],
                    )
                    st.plotly_chart(breakdown, use_container_width=True)
            else:
                try:
                    result = supervised.predict(user_text)
                    probabilities = result["probabilities"]
                    cols = st.columns(3)
                    cols[0].metric("Sentiment", result["label"])
                    cols[1].metric("Confidence", f"{float(result['score']):.1%}")
                    cols[2].metric("Entities", len(entities))

                    breakdown = go.Figure(
                        go.Bar(
                            x=["Positive", "Neutral", "Negative"],
                            y=[
                                probabilities.get("Positive", 0.0),
                                probabilities.get("Neutral", 0.0),
                                probabilities.get("Negative", 0.0),
                            ],
                            marker_color=["#16a34a", "#64748b", "#dc2626"],
                        )
                    )
                    breakdown.update_layout(
                        title="Supervised Probabilities",
                        height=300,
                        template="plotly_white",
                        yaxis_range=[0, 1],
                    )
                    st.plotly_chart(breakdown, use_container_width=True)
                except Exception as exc:
                    st.error(f"Supervised model failed: {exc}")

            st.subheader("Extracted Entities")
            if entity_df.empty:
                st.info("No named entities were detected in this review.")
            else:
                st.dataframe(entity_df, use_container_width=True, hide_index=True)

with tab_batch:
    st.subheader("Analyze Multiple Reviews")
    use_supervised = st.checkbox("Add supervised ML predictions", value=False)
    batch_input = st.text_area(
        "One review per line",
        height=220,
        placeholder="Great quality and fast delivery.\nRuns small compared with the size chart.\nThe design is beautiful but overpriced.",
    )

    if st.button("Analyze Batch", type="primary", key="batch_review_btn"):
        lines = [line.strip() for line in batch_input.splitlines() if line.strip()]
        if not lines:
            st.warning("Enter at least one review.")
        else:
            batch_df = pd.DataFrame({"text": lines})
            batch_df = vader.predict_batch(batch_df, text_col="text")
            batch_df = ner.predict_batch(batch_df, text_col="text")
            if use_supervised:
                try:
                    batch_df = supervised.predict_batch(batch_df, text_col="text")
                except Exception as exc:
                    st.warning(f"Supervised model skipped: {exc}")
            distribution = vader.get_sentiment_distribution(batch_df)

            cols = st.columns(3)
            cols[0].metric("Positive", f"{distribution['Positive']['count']} ({distribution['Positive']['percentage']}%)")
            cols[1].metric("Neutral", f"{distribution['Neutral']['count']} ({distribution['Neutral']['percentage']}%)")
            cols[2].metric("Negative", f"{distribution['Negative']['count']} ({distribution['Negative']['percentage']}%)")

            st.plotly_chart(Visualizer.sentiment_pie(distribution), use_container_width=True)
            display_columns = ["text", "vader_label", "vader_compound", "ner_entity_count", "ner_entity_text"]
            if use_supervised and "ml_label" in batch_df.columns:
                display_columns.extend(["ml_label", "ml_score"])
            st.dataframe(batch_df[display_columns], use_container_width=True, hide_index=True)

            st.download_button(
                "Download Batch Results",
                batch_df.to_csv(index=False).encode("utf-8"),
                "batch_review_analysis.csv",
                "text/csv",
                use_container_width=True,
            )

with tab_compare:
    st.subheader("Compare VADER, Supervised ML, and DistilBERT")
    compare_text = st.text_area(
        "Text to compare",
        height=120,
        placeholder="Try a nuanced review with mixed sentiment or sarcasm.",
        key="compare_text",
    )

    if st.button("Compare Models", type="primary", key="compare_models_btn"):
        if not compare_text.strip():
            st.warning("Enter a review to compare.")
        else:
            vader_result = vader.predict(compare_text)
            try:
                supervised_result = supervised.predict(compare_text)
                bert_result = BERTSentimentAnalyzer().predict(compare_text)
                comparison = pd.DataFrame(
                    [
                        {
                            "vader_label": vader_result["label"],
                            "ml_label": supervised_result["label"],
                            "bert_label": bert_result["label"],
                        }
                    ]
                )

                left, middle, right = st.columns(3)
                with left:
                    st.metric("VADER", vader_result["label"], help=f"Compound: {vader_result['compound']:.4f}")
                with middle:
                    st.metric(
                        "Supervised ML",
                        supervised_result["label"],
                        help=f"Confidence: {float(supervised_result['score']):.4f}",
                    )
                with right:
                    st.metric("DistilBERT", bert_result["label"], help=f"Confidence: {bert_result['score']:.4f}")

                labels = {vader_result["label"], supervised_result["label"], bert_result["label"]}
                if len(labels) == 1:
                    st.success("All models agree on this review.")
                else:
                    st.warning("The models disagree on this review. This is a good demo case for manual interpretation.")

                st.plotly_chart(Visualizer.bert_comparison(comparison), use_container_width=True)
            except Exception as exc:
                st.error(f"Model comparison failed: {exc}")