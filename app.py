"""
Streamlit entry point for the clothing factory sentiment analysis app.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Dict

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.aspect_analyzer import AspectSentimentAnalyzer
from models.decision_engine import DecisionEngine
from models.nlp_pipeline import NLPPipeline
from models.sentiment_bert import BERTSentimentAnalyzer
from models.topic_model import LDATopicModel
from utils.kpi_calculator import calculate_all_kpis
from utils.visualizations import Visualizer


SAMPLE_DATA_PATH = ROOT / "data" / "sample_reviews.csv"
REQUIRED_COLUMNS = ["text", "rating", "date", "category", "platform"]

COLUMN_ALIASES = {
    "text": [
        "text",
        "review",
        "reviews",
        "review_text",
        "review_body",
        "comment",
        "comments",
        "content",
        "feedback",
        "description",
        "review_description",
        "body",
        "customer_review",
        "review_title",
    ],
    "rating": [
        "rating",
        "stars",
        "score",
        "note",
        "grade",
        "review_rating",
        "star_rating",
        "overall_rating",
    ],
    "date": [
        "date",
        "review_date",
        "created_at",
        "timestamp",
        "time",
        "posted_at",
        "published_at",
        "submission_time",
    ],
    "category": [
        "category",
        "product",
        "product_category",
        "department",
        "department_name",
        "class",
        "class_name",
        "division_name",
        "product_type",
        "item_type",
        "segment",
    ],
    "platform": [
        "platform",
        "source",
        "marketplace",
        "site",
        "website",
        "channel",
        "vendor",
        "origin",
    ],
}


st.set_page_config(
    page_title="Clothing Review Intelligence",
    page_icon="🧵",
    layout="wide",
)


def load_css() -> None:
    css_path = ROOT / "assets" / "style.css"
    if css_path.exists():
        st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


def initialize_state() -> None:
    defaults = {
        "raw_df": None,
        "analysis_df": None,
        "aspect_df": pd.DataFrame(),
        "aspect_summary": pd.DataFrame(),
        "aspect_trends": pd.DataFrame(),
        "trend_directions": pd.DataFrame(),
        "topics": [],
        "topic_trends": pd.DataFrame(),
        "kpis": {},
        "decisions": [],
        "entity_summary": pd.DataFrame(),
        "preprocessing_report": {},
        "validation_report": {},
        "tfidf_shape": None,
        "embedding_shape": None,
        "ner_uses_spacy_model": False,
        "analysis_complete": False,
        "analysis_warnings": [],
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def reset_analysis() -> None:
    for key in [
        "analysis_df",
        "aspect_df",
        "aspect_summary",
        "aspect_trends",
        "trend_directions",
        "topics",
        "topic_trends",
        "kpis",
        "decisions",
        "entity_summary",
        "preprocessing_report",
        "validation_report",
        "tfidf_shape",
        "embedding_shape",
        "ner_uses_spacy_model",
        "analysis_warnings",
    ]:
        st.session_state.pop(key, None)
    st.session_state.analysis_complete = False


@st.cache_data(show_spinner=False)
def load_sample_data() -> pd.DataFrame:
    return pd.read_csv(SAMPLE_DATA_PATH)


def _normalize_header(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")


def _rename_known_columns(df: pd.DataFrame) -> pd.DataFrame:
    output = df.copy()
    output.columns = [_normalize_header(col) for col in output.columns]

    rename_map: dict[str, str] = {}
    for target, aliases in COLUMN_ALIASES.items():
        if target in output.columns:
            continue
        alias_set = {_normalize_header(alias) for alias in aliases}
        for column in output.columns:
            if column in alias_set and column not in rename_map:
                rename_map[column] = target
                break

    if rename_map:
        output = output.rename(columns=rename_map)
    return output


def normalize_reviews(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize uploaded review data with flexible column mapping."""
    output = _rename_known_columns(df)
    warnings: list[str] = []

    if "text" not in output.columns:
        raise ValueError(
            "No review text column found. Accepted names include: text, review, review_text, comment, content."
        )

    if "rating" not in output.columns:
        output["rating"] = pd.NA
        warnings.append("Rating column missing; rating-based KPIs and validation will be limited.")

    if "date" not in output.columns:
        output["date"] = pd.Timestamp.today().normalize()
        warnings.append("Date column missing; filled with today's date, so trend charts will reflect the upload date.")

    if "category" not in output.columns:
        output["category"] = "General"
        warnings.append("Category column missing; filled with 'General'.")

    if "platform" not in output.columns:
        output["platform"] = "Uploaded CSV"
        warnings.append("Platform column missing; filled with 'Uploaded CSV'.")

    output = output[REQUIRED_COLUMNS].copy()
    output["text"] = output["text"].fillna("").astype(str).str.strip()
    output = output[output["text"].str.len() > 0].reset_index(drop=True)
    if output.empty:
        raise ValueError("No usable review text found in the CSV.")

    output["rating"] = pd.to_numeric(output["rating"], errors="coerce")
    output["date"] = pd.to_datetime(output["date"], errors="coerce")
    output["category"] = output["category"].fillna("Unknown").astype(str).str.strip()
    output["platform"] = output["platform"].fillna("Unknown").astype(str).str.strip()

    if output["date"].isna().all():
        output["date"] = pd.Timestamp.today().normalize()
        warnings.append("Date values could not be parsed; filled with today's date.")

    if output["rating"].isna().all():
        warnings.append("No numeric ratings were found; rating-based KPIs and validation will be limited.")

    output["category"] = output["category"].replace("", "General")
    output["platform"] = output["platform"].replace("", "Uploaded CSV")

    output["review_id"] = range(1, len(output) + 1)
    output.attrs["normalization_warnings"] = warnings
    return output


def run_full_analysis(
    df: pd.DataFrame,
    use_distilbert: bool = False,
    n_topics: int = 5,
) -> Dict:
    """Run preprocessing, VADER, NER, optional DistilBERT, aspects, LDA, KPIs and decisions."""
    warnings: list[str] = []

    pipeline = NLPPipeline(n_topics=n_topics, topic_passes=12, embedding_dim=50)
    analyzed_df = pipeline.run(
        df,
        text_col="text",
        include_topics=False,
        include_validation=True,
    )

    if not pipeline.ner.uses_spacy_model:
        warnings.append("spaCy NER model unavailable; using the built-in regex fallback for entity extraction.")

    if use_distilbert:
        try:
            bert = BERTSentimentAnalyzer()
            analyzed_df = bert.predict_batch(analyzed_df, text_col="text")
        except Exception as exc:
            warnings.append(f"DistilBERT comparison skipped: {exc}")

    aspect_analyzer = AspectSentimentAnalyzer()
    analyzed_df, aspect_df = aspect_analyzer.analyze(analyzed_df, text_col="text")
    aspect_summary = aspect_analyzer.aspect_summary(aspect_df)
    aspect_trends = aspect_analyzer.trend_by_month(aspect_df)
    trend_directions = aspect_analyzer.trend_directions(aspect_trends)

    topics = []
    topic_trends = pd.DataFrame()
    try:
        topic_model = LDATopicModel(n_topics=n_topics, passes=12, min_df=1)
        topic_model.fit(analyzed_df["tokens"].tolist())
        analyzed_df = topic_model.assign_topics(analyzed_df, token_col="tokens")
        topics = topic_model.get_topics(n_words=8)
        topic_trends = topic_model.topic_trends(analyzed_df)
    except Exception as exc:
        analyzed_df["topic_label"] = "Unavailable"
        analyzed_df["topic_confidence"] = 0.0
        warnings.append(f"Topic modeling skipped: {exc}")

    validation_report = pipeline.validation_report or {}
    if validation_report.get("error"):
        warnings.append(f"Rating validation skipped: {validation_report['error']}")

    entity_summary = pd.DataFrame()
    if "ner_entities" in analyzed_df.columns:
        entity_summary = pipeline.ner.summarize_entities(analyzed_df)

    kpis = calculate_all_kpis(analyzed_df, aspect_df)
    decisions = DecisionEngine().generate(analyzed_df, aspect_df, topic_trends)

    return {
        "analysis_df": analyzed_df,
        "aspect_df": aspect_df,
        "aspect_summary": aspect_summary,
        "aspect_trends": aspect_trends,
        "trend_directions": trend_directions,
        "topics": topics,
        "topic_trends": topic_trends,
        "kpis": kpis,
        "decisions": decisions,
        "entity_summary": entity_summary,
        "preprocessing_report": pipeline.preprocessing_report or {},
        "validation_report": validation_report,
        "tfidf_shape": pipeline.tfidf_shape,
        "embedding_shape": pipeline.embedding_shape,
        "ner_uses_spacy_model": pipeline.ner.uses_spacy_model,
        "analysis_warnings": warnings,
    }


def save_analysis_results(results: Dict) -> None:
    for key, value in results.items():
        st.session_state[key] = value
    st.session_state.analysis_complete = True


def render_kpi_preview() -> None:
    kpis = st.session_state.get("kpis", {})
    if not kpis:
        return

    cols = st.columns(5)
    metric_keys = [
        "overall_customer_satisfaction",
        "quality_score",
        "return_risk_index",
        "delivery_score",
        "net_promoter_sentiment",
    ]
    for col, key in zip(cols, metric_keys):
        metric = kpis[key]
        value = metric["value"]
        suffix = metric.get("suffix", "")
        col.metric(metric["label"], f"{value:.1f}{suffix}", help=metric.get("help"))


def render_pipeline_snapshot() -> None:
    report = st.session_state.get("preprocessing_report", {})
    validation = st.session_state.get("validation_report", {})
    entity_summary = st.session_state.get("entity_summary", pd.DataFrame())
    tfidf_shape = st.session_state.get("tfidf_shape")
    embedding_shape = st.session_state.get("embedding_shape")
    analysis_df = st.session_state.get("analysis_df", pd.DataFrame())

    cols = st.columns(5)
    cols[0].metric("Rows After Cleaning", int(report.get("final_rows", 0)))
    cols[1].metric("Duplicates Removed", int(report.get("duplicates_removed", 0)))
    cols[2].metric("TF-IDF Features", int(tfidf_shape[1]) if tfidf_shape else 0)
    cols[3].metric("Embedding Size", int(embedding_shape[1]) if embedding_shape else 0)
    entity_total = int(analysis_df.get("ner_entity_count", pd.Series(dtype=int)).sum())
    cols[4].metric("Named Entities", entity_total)

    details = []
    if validation and not validation.get("error"):
        details.append(f"Rating validation accuracy: {validation.get('accuracy', 0.0):.2%}")
    details.append(
        "NER engine: spaCy model"
        if st.session_state.get("ner_uses_spacy_model")
        else "NER engine: regex fallback"
    )
    st.caption(" | ".join(details))

    if not entity_summary.empty:
        with st.expander("Top extracted entities", expanded=False):
            st.dataframe(entity_summary.head(10), use_container_width=True, hide_index=True)


load_css()
initialize_state()

st.title("Clothing Review Intelligence")
st.caption(
    "NLP pipeline for customer reviews: cleaning, deduplication, TF-IDF embeddings, NER, sentiment, aspects, "
    "LDA topics, KPIs, trends and business decisions."
)

with st.sidebar:
    st.header("Analysis Controls")
    use_distilbert = st.checkbox(
        "Compare with DistilBERT",
        value=False,
        help="Optional and slower. Downloads a HuggingFace model the first time it runs.",
    )
    n_topics = st.slider("LDA topics", min_value=2, max_value=8, value=5)
    st.divider()
    st.write("Required CSV columns")
    st.code("text, rating, date, category, platform", language="text")

left, right = st.columns([1.1, 0.9])

with left:
    st.subheader("1. Data Input")
    uploaded_file = st.file_uploader(
        "Upload customer reviews CSV",
        type=["csv"],
        help="The CSV must contain: text, rating, date, category and platform.",
    )

    col_upload, col_sample = st.columns(2)
    with col_upload:
        if uploaded_file is not None and st.button("Use Uploaded CSV", type="primary", use_container_width=True):
            try:
                normalized_df = normalize_reviews(pd.read_csv(uploaded_file))
                st.session_state.raw_df = normalized_df
                reset_analysis()
                st.success(f"Loaded {len(st.session_state.raw_df):,} uploaded reviews.")
                for warning in normalized_df.attrs.get("normalization_warnings", []):
                    st.info(warning)
            except Exception as exc:
                st.error(f"Could not load uploaded CSV: {exc}")

    with col_sample:
        if st.button("Load Sample Reviews", use_container_width=True):
            try:
                st.session_state.raw_df = normalize_reviews(load_sample_data())
                reset_analysis()
                st.success(f"Loaded {len(st.session_state.raw_df):,} sample clothing reviews.")
            except Exception as exc:
                st.error(f"Could not load sample data: {exc}")

with right:
    st.subheader("2. Run Analysis")
    st.write(
        "The pipeline classifies sentiment with VADER, detects clothing-specific aspects, "
        "discovers LDA topics, calculates business KPIs and generates decisions."
    )
    can_run = st.session_state.raw_df is not None
    if st.button("Run Full NLP Pipeline", disabled=not can_run, type="primary", use_container_width=True):
        with st.spinner("Analyzing reviews and generating recommendations..."):
            try:
                results = run_full_analysis(
                    st.session_state.raw_df,
                    use_distilbert=use_distilbert,
                    n_topics=n_topics,
                )
                save_analysis_results(results)
                st.success("Analysis complete. Open the pages from the sidebar for details.")
            except Exception as exc:
                st.session_state.analysis_complete = False
                st.error(f"Analysis failed: {exc}")

if st.session_state.raw_df is None:
    st.info("Load the sample clothing reviews or upload your own CSV to begin.")
else:
    st.divider()
    st.subheader("Loaded Review Preview")
    st.dataframe(st.session_state.raw_df.head(12), use_container_width=True)

if st.session_state.analysis_complete:
    st.divider()
    st.subheader("Analysis Snapshot")
    render_kpi_preview()

    st.subheader("Pipeline Snapshot")
    render_pipeline_snapshot()

    for warning in st.session_state.get("analysis_warnings", []):
        st.warning(warning)

    chart_left, chart_right = st.columns(2)
    with chart_left:
        st.plotly_chart(Visualizer.sentiment_pie(st.session_state.analysis_df), use_container_width=True)
    with chart_right:
        st.plotly_chart(
            Visualizer.aspect_sentiment_breakdown(st.session_state.aspect_df),
            use_container_width=True,
        )

    export_df = st.session_state.analysis_df.drop(columns=["topic_tokens"], errors="ignore")
    st.download_button(
        "Download analyzed reviews CSV",
        data=export_df.to_csv(index=False).encode("utf-8"),
        file_name="clothing_review_analysis.csv",
        mime="text/csv",
        use_container_width=True,
    )
