from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.visualizations import Visualizer


st.set_page_config(page_title="Aspect Sentiment", page_icon="🔎", layout="wide")

if not st.session_state.get("analysis_complete"):
    st.warning("Run the analysis from app.py first.")
    st.stop()

aspect_df = st.session_state.aspect_df
summary_df = st.session_state.aspect_summary

st.title("Aspect-Based Sentiment")
st.caption("Quality, sizing, delivery, price and design mentions extracted from each review.")

if aspect_df.empty:
    st.info("No configured aspect keywords were found in the loaded reviews.")
    st.stop()

left, right = st.columns([1.1, 0.9])
with left:
    st.plotly_chart(Visualizer.aspect_sentiment_breakdown(aspect_df), use_container_width=True)
with right:
    st.plotly_chart(Visualizer.aspect_score_bars(summary_df), use_container_width=True)

st.subheader("Aspect Summary")
st.dataframe(summary_df, use_container_width=True, hide_index=True)

st.subheader("Mention Explorer")
aspect_options = ["All"] + sorted(aspect_df["aspect"].unique().tolist())
sentiment_options = ["All", "Positive", "Neutral", "Negative"]
col1, col2 = st.columns(2)
selected_aspect = col1.selectbox("Aspect", aspect_options)
selected_sentiment = col2.selectbox("Sentiment", sentiment_options)

filtered = aspect_df.copy()
if selected_aspect != "All":
    filtered = filtered[filtered["aspect"] == selected_aspect]
if selected_sentiment != "All":
    filtered = filtered[filtered["sentiment"] == selected_sentiment]

display_cols = [
    "review_id",
    "aspect",
    "sentiment",
    "compound",
    "keywords",
    "context",
    "rating",
    "category",
    "platform",
    "date",
]
st.dataframe(filtered[display_cols], use_container_width=True, hide_index=True)
