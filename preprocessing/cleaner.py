"""
cleaner.py
----------
Full NLP preprocessing pipeline:
  1. Lowercasing
  2. URL / HTML / punctuation removal
  3. Stopword removal
  4. Optional lemmatization with spaCy
  5. Regex tokenization
  6. Dataset preparation with duplicate removal
"""

import re
import string
from typing import List, Optional, Tuple

import nltk
import pandas as pd
from nltk.corpus import stopwords


try:
    nltk.data.find("corpora/stopwords")
    STOPWORDS = set(stopwords.words("english"))
except Exception:
    STOPWORDS = {
        "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
        "has", "have", "i", "in", "is", "it", "its", "of", "on", "or", "that",
        "the", "this", "to", "was", "were", "with", "you", "your",
    }


class TextCleaner:
    """Configurable text cleaning pipeline for customer reviews."""

    def __init__(
        self,
        remove_stopwords: bool = True,
        lemmatize: bool = False,
        min_token_length: int = 2,
        extra_stopwords: Optional[List[str]] = None,
    ):
        self.remove_stopwords = remove_stopwords
        self.lemmatize = lemmatize
        self.min_token_length = min_token_length

        self.stopwords = STOPWORDS.copy()
        if extra_stopwords:
            self.stopwords.update(extra_stopwords)

        self._nlp = None

    def clean(self, text: str) -> str:
        """Return normalized text."""
        text = self._to_str(text)
        text = self._remove_urls(text)
        text = self._remove_html(text)
        text = self._remove_emojis(text)
        text = text.lower()
        text = self._remove_punctuation(text)
        return self._normalize_whitespace(text)

    def tokenize(self, text: str) -> List[str]:
        """Clean and tokenize text into modeling tokens."""
        cleaned = self.clean(text)
        # Keep simple word-like tokens (letters and apostrophes) for modeling.
        tokens = re.findall(r"[a-zA-Z][a-zA-Z']*", cleaned)
        tokens = [t for t in tokens if len(t) >= self.min_token_length]

        if self.remove_stopwords:
            tokens = [t for t in tokens if t not in self.stopwords]

        if self.lemmatize:
            tokens = self._lemmatize_tokens(tokens)

        return tokens

    def clean_series(self, series: pd.Series) -> pd.Series:
        """Apply cleaning to a pandas Series."""
        return series.apply(self.clean)

    def tokenize_series(self, series: pd.Series) -> pd.Series:
        """Apply tokenization to a pandas Series."""
        return series.apply(self.tokenize)

    def get_token_string(self, text: str) -> str:
        """Return tokenized text as a whitespace-joined string."""
        return " ".join(self.tokenize(text))

    def prepare_dataframe(
        self,
        df: pd.DataFrame,
        text_col: str = "text",
        drop_duplicates: bool = True,
    ) -> Tuple[pd.DataFrame, dict]:
        """
        Build a modeling-ready DataFrame.

        Adds:
            clean_text: normalized text
            tokens: token list
            token_text: whitespace-joined tokens for vectorizers/LDA
        """
        if text_col not in df.columns:
            raise ValueError(f"Column '{text_col}' not found.")

        prepared = df.copy()
        initial_rows = len(prepared)
        prepared = prepared.dropna(subset=[text_col])
        prepared[text_col] = prepared[text_col].astype(str).str.strip()
        prepared = prepared[prepared[text_col].str.len() > 0].copy()

        prepared["clean_text"] = self.clean_series(prepared[text_col])
        before_dedup = len(prepared)
        if drop_duplicates:
            prepared = prepared.drop_duplicates(subset=["clean_text"]).copy()
        after_dedup = len(prepared)

        prepared["tokens"] = self.tokenize_series(prepared[text_col])
        prepared["token_text"] = prepared["tokens"].apply(" ".join)
        prepared = prepared[prepared["tokens"].map(len) > 0].reset_index(drop=True)

        if prepared.empty:
            raise ValueError("No usable text remains after preprocessing.")

        report = {
            "initial_rows": int(initial_rows),
            "final_rows": int(len(prepared)),
            "removed_rows": int(initial_rows - len(prepared)),
            "duplicates_removed": int(before_dedup - after_dedup) if drop_duplicates else 0,
            "empty_token_rows_removed": int(after_dedup - len(prepared)),
            "text_column": text_col,
        }
        return prepared, report

    @staticmethod
    def _to_str(text) -> str:
        if pd.isna(text):
            return ""
        return str(text).strip()

    @staticmethod
    def _remove_urls(text: str) -> str:
        return re.sub(r"http\S+|www\.\S+", "", text)

    @staticmethod
    def _remove_html(text: str) -> str:
        return re.sub(r"<[^>]+>", "", text)

    @staticmethod
    def _remove_emojis(text: str) -> str:
        emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"
            "\U0001F300-\U0001F5FF"
            "\U0001F680-\U0001F6FF"
            "\U0001F1E0-\U0001F1FF"
            "\U00002702-\U000027B0"
            "\U000024C2-\U0001F251"
            "]+",
            flags=re.UNICODE,
        )
        return emoji_pattern.sub("", text)

    @staticmethod
    def _remove_punctuation(text: str) -> str:
        return text.translate(str.maketrans("", "", string.punctuation))

    @staticmethod
    def _normalize_whitespace(text: str) -> str:
        return re.sub(r"\s+", " ", text).strip()

    def _lemmatize_tokens(self, tokens: List[str]) -> List[str]:
        if self._nlp is None:
            try:
                import spacy

                self._nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
            except Exception:
                return tokens

        doc = self._nlp(" ".join(tokens))
        return [token.lemma_ for token in doc]
