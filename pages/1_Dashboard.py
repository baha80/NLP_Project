from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.visualizations import Visualizer


st.set_page_config(page_title="KPI Dashboard", page_icon="📊", layout="wide")


def require_analysis() -> None:
    if not st.session_state.get("analysis_complete"):
        st.warning("Run the analysis from app.py first.")
        st.stop()


require_analysis()

df = st.session_state.analysis_df
aspect_summary = st.session_state.aspect_summary
kpis = st.session_state.kpis
entity_summary = st.session_state.get("entity_summary")
ner_uses_spacy_model = st.session_state.get("ner_uses_spacy_model", False)

st.title("KPI Dashboard")
st.caption("Factory-facing indicators calculated from review sentiment, aspects and ratings.")

metric_keys = [
    "overall_customer_satisfaction",
    "quality_score",
    "return_risk_index",
    "delivery_score",
    "net_promoter_sentiment",
]

cols = st.columns(5)
for col, key in zip(cols, metric_keys):
    metric = kpis[key]
    value = metric["value"]
    suffix = metric.get("suffix", "")
    if key == "net_promoter_sentiment":
        col.metric(
            metric["label"],
            f"{value:.1f}{suffix}",
            help=f"{metric.get('help')} 5-star to 1-star ratio: {metric.get('ratio')}:1",
        )
    else:
        col.metric(metric["label"], f"{value:.1f}{suffix}", help=metric.get("help"))

st.divider()

left, right = st.columns([0.9, 1.1])
with left:
    st.plotly_chart(Visualizer.sentiment_pie(df), use_container_width=True)
with right:
    st.plotly_chart(Visualizer.sentiment_bar_by_category(df), use_container_width=True)

left2, right2 = st.columns(2)
with left2:
    st.plotly_chart(Visualizer.aspect_score_bars(aspect_summary), use_container_width=True)
with right2:
    st.plotly_chart(Visualizer.platform_sentiment(df), use_container_width=True)

st.subheader("NER Preview")
st.caption(
    "spaCy model active for entity extraction."
    if ner_uses_spacy_model
    else "spaCy model unavailable for this run, so the regex fallback extractor was used."
)

if entity_summary is None or entity_summary.empty:
    st.info("No entities were extracted from the current review set.")
else:
    entity_cols = st.columns(3)
    entity_cols[0].metric("Unique Entities", int(len(entity_summary)))
    entity_cols[1].metric("Entity Mentions", int(entity_summary["count"].sum()))
    reviews_with_entities = int((df.get("ner_entity_count", 0) > 0).sum()) if "ner_entity_count" in df.columns else 0
    entity_cols[2].metric("Reviews With Entities", reviews_with_entities)

    ner_left, ner_right = st.columns([0.9, 1.1])
    with ner_left:
        top_entities = entity_summary.head(8).sort_values("count", ascending=False)
        st.bar_chart(top_entities.set_index("entity")["count"], use_container_width=True)
    with ner_right:
        st.dataframe(entity_summary.head(10), use_container_width=True, hide_index=True)

st.subheader("Review Table")
display_cols = [
    "review_id",
    "text",
    "rating",
    "date",
    "category",
    "platform",
    "vader_label",
    "vader_compound",
    "topic_label",
]
display_cols = [col for col in display_cols if col in df.columns]
sentiment = st.selectbox("Filter sentiment", ["All", "Positive", "Neutral", "Negative"])
filtered = df if sentiment == "All" else df[df["vader_label"] == sentiment]
st.dataframe(filtered[display_cols], use_container_width=True, hide_index=True)
