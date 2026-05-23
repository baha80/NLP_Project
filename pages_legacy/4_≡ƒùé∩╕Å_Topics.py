"""
pages/4_🗂️_Topics.py
--------------------
LDA Topic Modeling page.
"""

import streamlit as st
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.topic_model import LDATopicModel
from preprocessing.cleaner import TextCleaner
from utils.visualizations import Visualizer

st.set_page_config(page_title="Topics", page_icon="🗂️", layout="wide")

css_path = os.path.join(os.path.dirname(__file__), "..", "assets", "style.css")
if os.path.exists(css_path):
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.markdown("# 🗂️ Topic Modeling — LDA")
st.markdown("Discover recurring themes across customer reviews using Latent Dirichlet Allocation.")
st.divider()

if "df" not in st.session_state or not st.session_state.get("analyzed", False):
    st.warning("⚠️ Please load and analyze data from the Home page first.")
    st.stop()

df = st.session_state.df

# ── LDA Configuration ─────────────────────────────────────────────────────────
st.markdown("### ⚙️ Configuration")

col1, col2, col3 = st.columns(3)
with col1:
    n_topics = st.slider("Number of Topics", min_value=2, max_value=10, value=5, step=1)
with col2:
    n_words = st.slider("Words per Topic", min_value=5, max_value=20, value=10, step=1)
with col3:
    passes = st.slider("Training Passes", min_value=5, max_value=30, value=10, step=5)

run_lda = st.button("🚀 Train LDA Model", type="primary")

if run_lda:
    with st.spinner("Preprocessing texts..."):
        cleaner = TextCleaner(remove_stopwords=True, lemmatize=False)
        token_lists = cleaner.tokenize_series(df["text"]).tolist()
        # Filter very short token lists
        token_lists = [t for t in token_lists if len(t) >= 3]

    with st.spinner(f"Training LDA with {n_topics} topics ({passes} passes)..."):
        lda_model = LDATopicModel(n_topics=n_topics, passes=passes)
        lda_model.fit(token_lists)

    st.session_state["lda_model"]   = lda_model
    st.session_state["lda_tokens"]  = token_lists

    with st.spinner("Assigning topics to reviews..."):
        df_with_topics = lda_model.assign_topics(df, text_col="text")
        st.session_state["df_topics"] = df_with_topics

    st.success(f"✅ LDA trained successfully with {n_topics} topics!")
    st.rerun()


# ── Results ────────────────────────────────────────────────────────────────────
if "lda_model" in st.session_state and st.session_state["lda_model"].is_fitted:
    lda_model    = st.session_state["lda_model"]
    df_topics    = st.session_state.get("df_topics", df)
    token_lists  = st.session_state.get("lda_tokens", [])

    topics = lda_model.get_topics(n_words=n_words)

    st.divider()

    # ── Topic overview ─────────────────────────────────────────────────────────
    st.markdown("### 📋 Discovered Topics")
    st.plotly_chart(Visualizer.topic_distribution(df_topics), use_container_width=True)

    # ── Topic cards ────────────────────────────────────────────────────────────
    st.markdown("### 🏷️ Topic Details")

    TOPIC_COLORS = ["#6366f1", "#06b6d4", "#10b981", "#f59e0b", "#ef4444", "#ec4899", "#8b5cf6", "#14b8a6", "#f97316", "#84cc16"]

    cols = st.columns(min(n_topics, 3))
    for i, topic in enumerate(topics):
        with cols[i % 3]:
            color = TOPIC_COLORS[i % len(TOPIC_COLORS)]
            words_badges = " ".join([
                f"<span style='background:{color}22;color:{color};padding:0.2rem 0.5rem;"
                f"border-radius:999px;font-size:0.8rem;margin:2px;display:inline-block;'>{w}</span>"
                for w in topic["words"][:8]
            ])
            count = len(df_topics[df_topics["topic_label"] == topic["label"]])

            st.markdown(
                f"<div style='background:rgba(99,102,241,0.08);border:1px solid {color}44;"
                f"border-radius:12px;padding:1.25rem;margin:0.5rem 0;'>"
                f"<div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:0.75rem;'>"
                f"<span style='font-weight:700;color:{color};font-size:1.1rem;'>{topic['label']}</span>"
                f"<span style='background:{color}22;color:{color};padding:0.2rem 0.6rem;border-radius:999px;font-size:0.85rem;'>{count} reviews</span>"
                f"</div>"
                f"<div>{words_badges}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

    st.divider()

    # ── Topic word charts ──────────────────────────────────────────────────────
    st.markdown("### 📊 Topic Word Distributions")

    selected_topic_label = st.selectbox(
        "Select topic to inspect",
        [t["label"] for t in topics],
    )
    selected_topic = next(t for t in topics if t["label"] == selected_topic_label)
    st.plotly_chart(Visualizer.topic_words_bar(selected_topic), use_container_width=True)

    st.divider()

    # ── Reviews by topic ──────────────────────────────────────────────────────
    st.markdown("### 📝 Reviews by Topic")

    topic_filter = st.selectbox(
        "Browse reviews from topic:",
        [t["label"] for t in topics],
        key="topic_browse",
    )

    filtered = df_topics[df_topics["topic_label"] == topic_filter][
        ["text", "topic_label", "topic_confidence", "vader_label"]
    ].sort_values("topic_confidence", ascending=False).head(10)

    st.dataframe(filtered.reset_index(drop=True), use_container_width=True)

    # ── Coherence score ────────────────────────────────────────────────────────
    st.divider()
    st.markdown("### 📐 Model Evaluation")

    if st.button("📐 Compute Coherence Score"):
        with st.spinner("Computing coherence score (C_v)..."):
            try:
                score = lda_model.coherence_score(token_lists)
                quality = "Excellent" if score > 0.55 else "Good" if score > 0.45 else "Fair"
                st.metric(
                    "Coherence Score (C_v)",
                    f"{score:.4f}",
                    delta=quality,
                )
                st.info(
                    "Coherence score ranges from 0 to 1. "
                    "Values > 0.55 indicate well-separated, meaningful topics."
                )
            except Exception as e:
                st.error(f"Could not compute coherence: {e}")

else:
    st.info("👆 Configure the parameters above and click **Train LDA Model** to start.")
