from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.visualizations import Visualizer


st.set_page_config(page_title="Topic Modeling", page_icon="🗂️", layout="wide")

if not st.session_state.get("analysis_complete"):
    st.warning("Run the analysis from app.py first.")
    st.stop()

df = st.session_state.analysis_df
topics = st.session_state.get("topics", [])
topic_trends = st.session_state.get("topic_trends", pd.DataFrame())

st.title("Topic Modeling Results")
st.caption("LDA discovers recurring themes and labels each topic with its strongest keywords.")

if not topics:
    st.info("Topic modeling was unavailable for this dataset. Check warnings on the upload page.")
    st.stop()

left, right = st.columns([1.0, 1.0])
with left:
    st.plotly_chart(Visualizer.topic_distribution(df), use_container_width=True)
with right:
    topic_labels = [topic["label"] for topic in topics]
    selected_label = st.selectbox("Topic keywords", topic_labels)
    selected_topic = next(topic for topic in topics if topic["label"] == selected_label)
    st.plotly_chart(Visualizer.topic_words_bar(selected_topic), use_container_width=True)

st.subheader("Topic Trend")
st.plotly_chart(Visualizer.topic_trend_area(topic_trends), use_container_width=True)

st.subheader("Reviews by Topic")
selected_review_topic = st.selectbox(
    "Filter topic",
    ["All"] + sorted(df["topic_label"].dropna().unique().tolist()),
)
filtered = df if selected_review_topic == "All" else df[df["topic_label"] == selected_review_topic]
display_cols = [
    "review_id",
    "text",
    "rating",
    "date",
    "category",
    "platform",
    "vader_label",
    "topic_label",
    "topic_confidence",
]
display_cols = [col for col in display_cols if col in filtered.columns]
st.dataframe(filtered[display_cols], use_container_width=True, hide_index=True)
