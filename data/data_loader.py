"""
data_loader.py
--------------
Handles loading data from:
- Local CSV files
- Kaggle datasets (via kaggle API)
- Sample built-in data
"""

import pandas as pd
import os
from pathlib import Path

SAMPLE_DATA_PATH = Path(__file__).parent / "sample_reviews.csv"
REVIEWS_DATA_PATH = Path(__file__).parent / "reviews.csv"


def load_sample_data() -> pd.DataFrame:
    """Load the built-in sample dataset."""
    df = pd.read_csv(SAMPLE_DATA_PATH)
    df["date"] = pd.to_datetime(df["date"])
    return _normalize_columns(df)


def load_reviews_data() -> pd.DataFrame:
    """Load the main project reviews dataset."""
    df = pd.read_csv(REVIEWS_DATA_PATH)
    return _normalize_columns(df)


def load_csv(file) -> pd.DataFrame:
    """
    Load a user-uploaded CSV file from Streamlit file uploader.

    Expected columns (flexible):
        - text / review / comment / content  → review text
        - rating / stars / score             → numeric rating (optional)
        - date / created_at                  → date (optional)
        - platform / source                  → source platform (optional)
        - category / product                 → category (optional)

    Returns a normalized DataFrame with column: 'text'
    """
    df = pd.read_csv(file)
    df = _normalize_columns(df)
    return df


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename common column variants to standard names."""
    rename_map = {}

    text_candidates = ["review", "comment", "content", "feedback", "description"]
    for col in df.columns:
        if col.lower() in text_candidates and "text" not in df.columns:
            rename_map[col] = "text"

    rating_candidates = ["stars", "score", "note", "grade"]
    for col in df.columns:
        if col.lower() in rating_candidates and "rating" not in df.columns:
            rename_map[col] = "rating"

    date_candidates = ["created_at", "timestamp", "time", "posted_at"]
    for col in df.columns:
        if col.lower() in date_candidates and "date" not in df.columns:
            rename_map[col] = "date"

    df = df.rename(columns=rename_map)

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    if "text" not in df.columns:
        raise ValueError(
            "No text column found. Please ensure your CSV has a column named "
            "'text', 'review', 'comment', or 'content'."
        )

    df = df.dropna(subset=["text"])
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"].str.len() > 10]
    df = df.reset_index(drop=True)

    return df
