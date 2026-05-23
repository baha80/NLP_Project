"""
pages/2_🔍_Analyze.py
---------------------
Real-time text sentiment analysis page.
Supports: single text, batch text, and DistilBERT comparison.
"""

import streamlit as st
import pandas as pd
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.sentiment_vader import VADERSentimentAnalyzer
from utils.visualizations import Visualizer

st.set_page_config(page_title="Analyze", page_icon="🔍", layout="wide")

css_path = os.path.join(os.path.dirname(__file__), "..", "assets", "style.css")
if os.path.exists(css_path):
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.markdown("# 🔍 Real-time Sentiment Analysis")
st.divider()

vader = VADERSentimentAnalyzer()

tab1, tab2, tab3 = st.tabs(["✏️ Single Text", "📋 Multi-Text", "🤖 BERT Comparison"])

# ── Tab 1: Single text analysis ───────────────────────────────────────────────
with tab1:
    st.markdown("### Analyze a single review")

    sample_texts = [
        "Select an example or type your own...",
        "I absolutely love this product! The quality is amazing and delivery was super fast.",
        "Terrible experience. The item arrived broken and customer service was completely unhelpful.",
        "It's okay, nothing special. Does what it says but nothing more.",
        "Best purchase I've made this year! Highly recommend to everyone.",
        "Very disappointed. The product looks nothing like the photos.",
    ]
    selected = st.selectbox("Quick examples", sample_texts)

    default_text = "" if selected == sample_texts[0] else selected
    user_text = st.text_area(
        "Enter review text",
        value=default_text,
        height=120,
        placeholder="Type or paste a customer review here...",
    )

    col_analyze, col_clear = st.columns([1, 5])
    analyze_clicked = col_analyze.button("⚡ Analyze", type="primary")
    if col_clear.button("🗑 Clear"):
        user_text = ""

    if analyze_clicked and user_text.strip():
        result = vader.predict(user_text)

        # Sentiment label with color
        label_colors = {
            "Positive": ("#22c55e", "😊"),
            "Negative": ("#ef4444", "😞"),
            "Neutral":  ("#94a3b8", "😐"),
        }
        color, emoji = label_colors.get(result["label"], ("#94a3b8", "😐"))

        st.markdown("---")
        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(
            f"<div style='text-align:center;padding:1rem;background:rgba(99,102,241,0.1);"
            f"border-radius:12px;border:1px solid {color}44;'>"
            f"<div style='font-size:2rem'>{emoji}</div>"
            f"<div style='font-size:1.2rem;font-weight:700;color:{color};'>{result['label']}</div>"
            f"<div style='color:#94a3b8;font-size:0.85rem;'>Sentiment</div></div>",
            unsafe_allow_html=True,
        )
        c2.metric("Compound Score", f"{result['compound']:.4f}")
        c3.metric("Confidence",     f"{result['confidence']:.1%}")
        c4.metric("Positive Score", f"{result['positive']:.4f}")

        st.divider()
        col_gauge, col_scores = st.columns([1, 1])
        with col_gauge:
            st.plotly_chart(Visualizer.sentiment_gauge(result["compound"]), use_container_width=True)
        with col_scores:
            import plotly.graph_objects as go
            fig = go.Figure(go.Bar(
                x=["Positive", "Neutral", "Negative"],
                y=[result["positive"], result["neutral"], result["negative"]],
                marker_color=["#22c55e", "#94a3b8", "#ef4444"],
            ))
            fig.update_layout(
                title="Score Breakdown",
                height=300,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                yaxis_range=[0, 1],
            )
            st.plotly_chart(fig, use_container_width=True)

    elif analyze_clicked:
        st.warning("⚠️ Please enter some text to analyze.")


# ── Tab 2: Multi-text batch ────────────────────────────────────────────────────
with tab2:
    st.markdown("### Analyze multiple reviews at once")
    st.markdown("Enter one review per line:")

    batch_input = st.text_area(
        "Batch input",
        height=200,
        placeholder="Paste multiple reviews here, one per line...\n\nGreat product!\nAwful experience.\nIt was okay.",
    )

    if st.button("⚡ Analyze All", type="primary", key="batch_btn"):
        lines = [l.strip() for l in batch_input.split("\n") if l.strip()]
        if lines:
            batch_df = pd.DataFrame({"text": lines})
            batch_df = vader.predict_batch(batch_df, text_col="text")

            st.success(f"✅ Analyzed {len(batch_df)} reviews")

            # Summary
            dist = vader.get_sentiment_distribution(batch_df)
            c1, c2, c3 = st.columns(3)
            c1.metric("😊 Positive", f"{dist['Positive']['count']} ({dist['Positive']['percentage']}%)")
            c2.metric("😞 Negative", f"{dist['Negative']['count']} ({dist['Negative']['percentage']}%)")
            c3.metric("😐 Neutral",  f"{dist['Neutral']['count']} ({dist['Neutral']['percentage']}%)")

            st.divider()
            st.plotly_chart(Visualizer.sentiment_pie(dist), use_container_width=True)

            display_df = batch_df[["text", "vader_label", "vader_compound", "vader_confidence"]].copy()
            display_df.columns = ["Review", "Sentiment", "Score", "Confidence"]
            st.dataframe(display_df, use_container_width=True)

            csv = batch_df.to_csv(index=False).encode()
            st.download_button("⬇️ Download Results", csv, "batch_results.csv", "text/csv")
        else:
            st.warning("⚠️ Please enter at least one review.")


# ── Tab 3: BERT comparison ────────────────────────────────────────────────────
with tab3:
    st.markdown("### 🤖 Compare VADER vs DistilBERT")
    st.info(
        "DistilBERT provides higher accuracy but requires ~500MB model download on first use. "
        "It uses deep contextual embeddings vs VADER's rule-based approach."
    )

    bert_text = st.text_area(
        "Enter text to compare",
        height=100,
        placeholder="Try something nuanced, like sarcasm...",
        key="bert_input",
    )

    if st.button("🤖 Compare Models", type="primary"):
        if bert_text.strip():
            # VADER result (instant)
            vader_result = vader.predict(bert_text)

            with st.spinner("Loading DistilBERT model (first run may take 1-2 min)..."):
                try:
                    from models.sentiment_bert import BERTSentimentAnalyzer
                    bert_analyzer = BERTSentimentAnalyzer()
                    bert_result = bert_analyzer.predict(bert_text)
                    bert_available = True
                except Exception as e:
                    bert_available = False
                    bert_error = str(e)

            col_vader, col_bert = st.columns(2)

            with col_vader:
                st.markdown("#### VADER (Rule-based)")
                label_color = {"Positive": "#22c55e", "Negative": "#ef4444", "Neutral": "#94a3b8"}
                color = label_color.get(vader_result["label"], "#94a3b8")
                st.markdown(
                    f"<div style='background:rgba(99,102,241,0.1);border-radius:12px;"
                    f"border:1px solid {color}44;padding:1.5rem;text-align:center;'>"
                    f"<div style='font-size:2rem;font-weight:800;color:{color};'>{vader_result['label']}</div>"
                    f"<div style='color:#94a3b8;'>Score: {vader_result['compound']:.4f}</div></div>",
                    unsafe_allow_html=True,
                )

            with col_bert:
                st.markdown("#### DistilBERT (Transformer)")
                if bert_available:
                    color = label_color.get(bert_result["label"], "#94a3b8")
                    st.markdown(
                        f"<div style='background:rgba(6,182,212,0.1);border-radius:12px;"
                        f"border:1px solid {color}44;padding:1.5rem;text-align:center;'>"
                        f"<div style='font-size:2rem;font-weight:800;color:{color};'>{bert_result['label']}</div>"
                        f"<div style='color:#94a3b8;'>Confidence: {bert_result['score']:.4f}</div></div>",
                        unsafe_allow_html=True,
                    )
                    if vader_result["label"] == bert_result["label"]:
                        st.success("✅ Both models agree!")
                    else:
                        st.warning("⚠️ Models disagree — consider reviewing manually.")
                else:
                    st.error(f"BERT unavailable: {bert_error}")
        else:
            st.warning("⚠️ Please enter text to analyze.")
