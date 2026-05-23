from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.visualizations import Visualizer


st.set_page_config(page_title="Trend Analysis", page_icon="📈", layout="wide")

if not st.session_state.get("analysis_complete"):
    st.warning("Run the analysis from app.py first.")
    st.stop()

df = st.session_state.analysis_df
aspect_trends = st.session_state.get("aspect_trends", pd.DataFrame())
trend_directions = st.session_state.get("trend_directions", pd.DataFrame())
topic_trends = st.session_state.get("topic_trends", pd.DataFrame())

st.title("Trend Analysis")
st.caption("Monthly sentiment movement by review and by clothing aspect.")

st.plotly_chart(Visualizer.monthly_sentiment_line(df), use_container_width=True)

st.subheader("Aspect Trend")
if aspect_trends.empty:
    st.info("No aspect trend data is available.")
else:
    aspects = sorted(aspect_trends["aspect"].unique().tolist())
    col1, col2 = st.columns([1.2, 0.8])
    selected_aspects = col1.multiselect("Aspects", aspects, default=aspects)
    metric = col2.selectbox(
        "Metric",
        ["positive_rate", "negative_rate", "mentions"],
        format_func=lambda value: value.replace("_", " ").title(),
    )
    st.plotly_chart(
        Visualizer.aspect_trend_line(aspect_trends, metric=metric, aspects=selected_aspects),
        use_container_width=True,
    )

    st.subheader("Improving or Declining")
    st.dataframe(trend_directions, use_container_width=True, hide_index=True)

st.subheader("Topic Trend")
st.plotly_chart(Visualizer.topic_trend_area(topic_trends), use_container_width=True)
