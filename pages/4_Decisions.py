from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.visualizations import Visualizer


st.set_page_config(page_title="Decision Recommendations", page_icon="✅", layout="wide")

if not st.session_state.get("analysis_complete"):
    st.warning("Run the analysis from app.py first.")
    st.stop()

decisions = st.session_state.get("decisions", [])

st.title("Decision Recommendations")
st.caption("Automatic business actions with priority levels based on thresholds and topic spikes.")

if not decisions:
    st.info("No recommendations were generated.")
    st.stop()

left, right = st.columns([0.8, 1.2])
with left:
    st.plotly_chart(Visualizer.decision_priority_bar(decisions), use_container_width=True)
with right:
    priority_filter = st.multiselect(
        "Priority filter",
        ["🚨 Critical", "⚠️ Warning", "✅ Good"],
        default=["🚨 Critical", "⚠️ Warning", "✅ Good"],
    )
    filtered = [item for item in decisions if item["priority"] in priority_filter]

    for item in filtered:
        message = (
            f"**{item['priority']} - {item['title']}**  \n"
            f"{item['recommendation']}  \n"
            f"Metric: {item['metric_value']:.1f}% | Threshold: {item['threshold']:.1f}% | Category: {item['category']}"
        )
        if item["priority"] == "🚨 Critical":
            st.error(message)
        elif item["priority"] == "⚠️ Warning":
            st.warning(message)
        else:
            st.success(message)

st.subheader("Decision Table")
st.dataframe(pd.DataFrame(decisions), use_container_width=True, hide_index=True)
