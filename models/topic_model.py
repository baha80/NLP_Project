"""
LDA topic modeling with Gensim.

Topics are used to discover recurring complaint and praise themes in customer
reviews, then assign each review to its strongest topic.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd


class LDATopicModel:
    """Small wrapper around Gensim LDA for Streamlit-friendly outputs."""

    EXTRA_STOPWORDS = {
        "item",
        "product",
        "order",
        "ordered",
        "really",
        "just",
        "would",
        "could",
        "also",
        "got",
        "get",
        "wear",
        "wore",
        "clothes",
        "clothing",
    }

    def __init__(
        self,
        n_topics: int = 5,
        passes: int = 12,
        random_state: int = 42,
        min_df: int = 1,
    ):
        self.n_topics = n_topics
        self.passes = passes
        self.random_state = random_state
        self.min_df = min_df
        self._lda = None
        self._dictionary = None
        self._corpus = None
        self._actual_topics = 0

    def preprocess_text(self, text: str) -> List[str]:
        """Clean and tokenize review text for topic modeling."""
        from gensim.parsing.preprocessing import STOPWORDS
        from gensim.utils import simple_preprocess

        # Combine Gensim stopwords with domain-specific terms.
        stopwords = set(STOPWORDS).union(self.EXTRA_STOPWORDS)
        tokens = simple_preprocess(str(text), deacc=True, min_len=3)
        return [token for token in tokens if token not in stopwords]

    def prepare_tokens(self, texts: Iterable[str]) -> List[List[str]]:
        return [self.preprocess_text(text) for text in texts]

    def fit(self, token_lists: List[List[str]]) -> "LDATopicModel":
        """Train the LDA model."""
        from gensim import corpora
        from gensim.models import LdaModel

        clean_tokens = [tokens for tokens in token_lists if tokens]
        if not clean_tokens:
            raise ValueError("Topic modeling needs at least one non-empty tokenized review.")

        self._dictionary = corpora.Dictionary(clean_tokens)
        if len(clean_tokens) >= 5:
            # Remove rare and overly common terms for more stable topics.
            self._dictionary.filter_extremes(no_below=self.min_df, no_above=0.85)

        if len(self._dictionary) == 0:
            self._dictionary = corpora.Dictionary(clean_tokens)

        self._corpus = [self._dictionary.doc2bow(tokens) for tokens in clean_tokens]
        self._corpus = [doc for doc in self._corpus if doc]
        if not self._corpus:
            raise ValueError("Topic modeling dictionary is empty after preprocessing.")

        self._actual_topics = min(self.n_topics, len(self._dictionary), len(self._corpus))
        self._actual_topics = max(self._actual_topics, 1)

        self._lda = LdaModel(
            corpus=self._corpus,
            id2word=self._dictionary,
            num_topics=self._actual_topics,
            passes=self.passes,
            random_state=self.random_state,
            alpha="auto",
            eta="auto",
        )
        return self

    def fit_transform(
        self,
        df: pd.DataFrame,
        text_col: str = "text",
        token_col: str = "topic_tokens",
    ) -> pd.DataFrame:
        """Fit LDA on the DataFrame and assign a dominant topic per review."""
        output = df.copy().reset_index(drop=True)
        output[token_col] = self.prepare_tokens(output[text_col].fillna("").astype(str))
        self.fit(output[token_col].tolist())
        return self.assign_topics(output, token_col=token_col)

    def get_topics(self, n_words: int = 8) -> List[Dict]:
        """Return topic labels and top keywords."""
        self._check_fitted()
        topics = []
        for topic_id in range(self._actual_topics):
            top_words = self._lda.show_topic(topic_id, topn=n_words)
            words = [word for word, _ in top_words]
            weights = [round(float(weight), 4) for _, weight in top_words]
            readable = ", ".join(words[:3]).title() if words else "Unlabeled"
            topics.append(
                {
                    "topic_id": topic_id,
                    "label": f"Topic {topic_id + 1}: {readable}",
                    "words": words,
                    "weights": weights,
                    "representation": " + ".join(words[:5]),
                }
            )
        return topics

    def get_dominant_topic(self, tokens: List[str]) -> Tuple[int, float]:
        """Return dominant topic id and probability for one tokenized review."""
        self._check_fitted()
        bow = self._dictionary.doc2bow(tokens)
        if not bow:
            return -1, 0.0

        topic_probs = self._lda.get_document_topics(bow, minimum_probability=0.0)
        if not topic_probs:
            return -1, 0.0

        topic_id, probability = max(topic_probs, key=lambda item: item[1])
        return int(topic_id), round(float(probability), 4)

    def assign_topics(
        self,
        df: pd.DataFrame,
        token_col: str = "topic_tokens",
        text_col: Optional[str] = None,
    ) -> pd.DataFrame:
        """Add dominant topic columns to a DataFrame."""
        self._check_fitted()
        output = df.copy().reset_index(drop=True)

        if token_col not in output.columns:
            if text_col is None:
                raise ValueError(f"Column '{token_col}' not found. Pass text_col.")
            output[token_col] = self.prepare_tokens(output[text_col].fillna("").astype(str))

        dominant = output[token_col].apply(self.get_dominant_topic)
        label_map = {topic["topic_id"]: topic["label"] for topic in self.get_topics()}
        output["topic_id"] = [topic_id for topic_id, _ in dominant]
        output["topic_confidence"] = [confidence for _, confidence in dominant]
        output["topic_label"] = output["topic_id"].map(label_map).fillna("Unknown")
        return output

    def topic_trends(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return monthly topic counts and shares."""
        if "date" not in df.columns or "topic_label" not in df.columns:
            return pd.DataFrame()

        data = df.copy()
        data["date"] = pd.to_datetime(data["date"], errors="coerce")
        data = data.dropna(subset=["date"])
        if data.empty:
            return pd.DataFrame()

        data["month"] = data["date"].dt.to_period("M").astype(str)
        counts = data.groupby(["month", "topic_label"]).size().reset_index(name="count")
        totals = counts.groupby("month")["count"].transform("sum")
        counts["share"] = (counts["count"] / totals * 100).round(1)
        return counts.sort_values(["month", "count"], ascending=[True, False])

    def _check_fitted(self) -> None:
        if self._lda is None or self._dictionary is None:
            raise RuntimeError("Topic model is not fitted. Call fit() first.")

    @property
    def is_fitted(self) -> bool:
        return self._lda is not None
