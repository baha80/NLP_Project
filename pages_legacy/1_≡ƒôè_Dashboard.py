"""
pages/1_📊_Dashboard.py
-----------------------
Full dashboard with all KPIs and charts.
"""

import streamlit as st
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.visualizations import Visualizer
from utils.metrics import model_comparison_chart, agreement_rate
from models.sentiment_vader import VADERSentimentAnalyzer

st.set_page_config(page_title="Dashboard", page_icon="📊", layout="wide")

# Load CSS
css_path = os.path.join(os.path.dirname(__file__), "..", "assets", "style.css")
if os.path.exists(css_path):
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.markdown("# 📊 Dashboard")
st.markdown("Complete overview of sentiment analysis results.")
st.divider()

if "df" not in st.session_state or st.session_state.df is None:
    st.warning("⚠️ No data loaded. Please go to the Home page and load a dataset.")
    st.stop()

if not st.session_state.get("analyzed", False):
    st.warning("⚠️ Data not yet analyzed. Please run the analysis from the Home page.")
    st.stop()

df = st.session_state.df
analyzer = VADERSentimentAnalyzer()
distribution = analyzer.get_sentiment_distribution(df)

# ── KPI row ────────────────────────────────────────────────────────────────
total   = len(df)
avg_pos = distribution["Positive"]["percentage"]
avg_neg = distribution["Negative"]["percentage"]
avg_cmp = df["vader_compound"].mean()

cols = st.columns(4)
cols[0].metric("Total Reviews", f"{total:,}")
cols[1].metric("😊 Positive Rate", f"{avg_pos}%")
cols[2].metric("😞 Negative Rate", f"{avg_neg}%")
cols[3].metric("Avg Compound", f"{avg_cmp:.3f}")

st.divider()

# ── Tabs ────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📊 Distribution", "📈 Trends", "🔬 Model Comparison"])

with tab1:
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(Visualizer.sentiment_pie(distribution), use_container_width=True)
    with c2:
        st.plotly_chart(Visualizer.sentiment_bar(df), use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        st.plotly_chart(Visualizer.compound_histogram(df), use_container_width=True)
    with c4:
        if "rating" in df.columns:
            st.plotly_chart(Visualizer.rating_distribution(df), use_container_width=True)

with tab2:
    if "date" in df.columns:
        st.plotly_chart(Visualizer.sentiment_over_time(df), use_container_width=True)
    else:
        st.info("No date column found in the dataset. Temporal analysis unavailable.")

with tab3:
    if "bert_label" in df.columns:
        agree = agreement_rate(df, "vader_label", "bert_label")
        st.metric("Model Agreement Rate", f"{agree}%")
        st.plotly_chart(model_comparison_chart(df), use_container_width=True)
    else:
        st.info("Run DistilBERT analysis from the Analyze page to enable model comparison.")
