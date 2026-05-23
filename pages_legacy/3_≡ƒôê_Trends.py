"""
pages/3_📈_Trends.py
--------------------
Temporal trends + WordCloud analysis page.
"""

import streamlit as st
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.visualizations import Visualizer
import base64

st.set_page_config(page_title="Trends", page_icon="📈", layout="wide")

css_path = os.path.join(os.path.dirname(__file__), "..", "assets", "style.css")
if os.path.exists(css_path):
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.markdown("# 📈 Trends & Word Analysis")
st.divider()

if "df" not in st.session_state or not st.session_state.get("analyzed", False):
    st.warning("⚠️ Please load and analyze data from the Home page first.")
    st.stop()

df = st.session_state.df

# ── Temporal trends ────────────────────────────────────────────────────────────
st.markdown("### 📅 Temporal Analysis")

if "date" in df.columns and df["date"].notna().any():
    st.plotly_chart(Visualizer.sentiment_over_time(df), use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        if "rating" in df.columns:
            st.plotly_chart(Visualizer.rating_distribution(df), use_container_width=True)
    with col2:
        st.plotly_chart(Visualizer.compound_histogram(df), use_container_width=True)
else:
    st.info("ℹ️ No date column found. Showing score distribution instead.")
    st.plotly_chart(Visualizer.compound_histogram(df), use_container_width=True)

st.divider()

# ── Platform analysis ──────────────────────────────────────────────────────────
if "platform" in df.columns:
    st.markdown("### 🌐 Analysis by Platform")
    col1, col2 = st.columns(2)

    platform_counts = df["platform"].value_counts()
    import plotly.express as px

    with col1:
        fig_plat = px.pie(
            values=platform_counts.values,
            names=platform_counts.index,
            title="Reviews by Platform",
            color_discrete_sequence=["#6366f1", "#06b6d4", "#10b981", "#f59e0b"],
        )
        fig_plat.update_layout(height=350, paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_plat, use_container_width=True)

    with col2:
        platform_sentiment = df.groupby(["platform", "vader_label"]).size().reset_index(name="count")
        fig_ps = px.bar(
            platform_sentiment,
            x="platform", y="count", color="vader_label",
            color_discrete_map={"Positive": "#22c55e", "Negative": "#ef4444", "Neutral": "#94a3b8"},
            title="Sentiment by Platform",
            barmode="group",
        )
        fig_ps.update_layout(height=350, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_ps, use_container_width=True)

    st.divider()

# ── WordCloud ──────────────────────────────────────────────────────────────────
st.markdown("### ☁️ Word Cloud")

sentiment_filter = st.selectbox(
    "Generate WordCloud for:",
    ["all", "Positive", "Negative", "Neutral"],
    format_func=lambda x: {"all": "🌈 All Reviews"}.get(x, f"{'😊' if x=='Positive' else '😞' if x=='Negative' else '😐'} {x}"),
)

if st.button("🎨 Generate Word Cloud", type="primary"):
    if sentiment_filter == "all":
        texts = df["text"].tolist()
    else:
        texts = df[df["vader_label"] == sentiment_filter]["text"].tolist()

    if texts:
        with st.spinner("Generating word cloud..."):
            img_b64 = Visualizer.wordcloud_image(texts, sentiment=sentiment_filter)

        if img_b64:
            st.image(
                f"data:image/png;base64,{img_b64}",
                use_column_width=True,
                caption=f"Word Cloud — {sentiment_filter.upper()} reviews ({len(texts)} texts)",
            )
        else:
            st.warning("Could not generate WordCloud. Make sure 'wordcloud' is installed.")
    else:
        st.warning("No reviews found for the selected filter.")

st.divider()

# ── Top reviews ────────────────────────────────────────────────────────────────
st.markdown("### 🏆 Extreme Reviews")
col_best, col_worst = st.columns(2)

with col_best:
    st.markdown("#### 😊 Most Positive")
    top_pos = df.nlargest(5, "vader_compound")[["text", "vader_compound"]].reset_index(drop=True)
    for _, row in top_pos.iterrows():
        st.markdown(
            f"<div style='background:rgba(34,197,94,0.1);border-left:3px solid #22c55e;"
            f"padding:0.75rem;border-radius:0 8px 8px 0;margin:0.5rem 0;font-size:0.9rem;'>"
            f"{row['text'][:150]}... <span style='color:#22c55e;font-weight:700;'>({row['vader_compound']:.3f})</span></div>",
            unsafe_allow_html=True,
        )

with col_worst:
    st.markdown("#### 😞 Most Negative")
    top_neg = df.nsmallest(5, "vader_compound")[["text", "vader_compound"]].reset_index(drop=True)
    for _, row in top_neg.iterrows():
        st.markdown(
            f"<div style='background:rgba(239,68,68,0.1);border-left:3px solid #ef4444;"
            f"padding:0.75rem;border-radius:0 8px 8px 0;margin:0.5rem 0;font-size:0.9rem;'>"
            f"{row['text'][:150]}... <span style='color:#ef4444;font-weight:700;'>({row['vader_compound']:.3f})</span></div>",
            unsafe_allow_html=True,
        )
