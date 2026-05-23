from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


st.set_page_config(page_title="NER Explorer", page_icon="🏷️", layout="wide")


def load_css() -> None:
    css_path = ROOT / "assets" / "style.css"
    if css_path.exists():
        st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


def has_entity(entities: list[dict], target: str) -> bool:
    return any(item.get("text") == target for item in entities)


def has_label(entities: list[dict], target: str) -> bool:
    return any(item.get("label") == target for item in entities)


load_css()

if not st.session_state.get("analysis_complete"):
    st.warning("Run the analysis from app.py first.")
    st.stop()

df = st.session_state.analysis_df
entity_summary = st.session_state.get("entity_summary", pd.DataFrame())

st.title("Named Entity Explorer")
st.caption("Inspect the brands, locations, organizations, and other entities extracted from the cleaned review corpus.")

if st.session_state.get("ner_uses_spacy_model"):
    st.success("spaCy entity model is active for this analysis run.")
else:
    st.info("spaCy model was unavailable, so the app used the built-in regex fallback for entity extraction.")

if entity_summary.empty:
    st.info("No entities were extracted from the current review set.")
    st.stop()

entity_mentions = int(entity_summary["count"].sum())
reviews_with_entities = int((df.get("ner_entity_count", pd.Series(dtype=int)) > 0).sum())

cols = st.columns(3)
cols[0].metric("Unique Entities", int(len(entity_summary)))
cols[1].metric("Entity Mentions", entity_mentions)
cols[2].metric("Reviews With Entities", reviews_with_entities)

left, right = st.columns([0.9, 1.1])
with left:
    st.subheader("Top Entities")
    st.dataframe(entity_summary.head(20), use_container_width=True, hide_index=True)
with right:
    chart_df = entity_summary.head(10).sort_values("count", ascending=True)
    fig = px.bar(
        chart_df,
        x="count",
        y="entity",
        color="label",
        orientation="h",
        title="Most Frequent Extracted Entities",
        labels={"count": "Mentions", "entity": "Entity", "label": "Label"},
    )
    fig.update_layout(height=420, template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

st.subheader("Entity Review Explorer")
label_options = ["All"] + sorted(entity_summary["label"].dropna().astype(str).unique().tolist())
entity_options = ["All"] + entity_summary["entity"].astype(str).head(50).tolist()

filter_left, filter_right = st.columns(2)
selected_label = filter_left.selectbox("Entity label", label_options)
selected_entity = filter_right.selectbox("Entity text", entity_options)

filtered = df.copy()
if selected_label != "All":
    filtered = filtered[filtered["ner_entities"].apply(lambda items: has_label(items, selected_label))]
if selected_entity != "All":
    filtered = filtered[filtered["ner_entities"].apply(lambda items: has_entity(items, selected_entity))]

display_cols = [
    "review_id",
    "text",
    "ner_entity_count",
    "ner_entity_text",
    "vader_label",
    "category",
    "platform",
    "date",
]
display_cols = [col for col in display_cols if col in filtered.columns]
st.dataframe(filtered[display_cols], use_container_width=True, hide_index=True)