"""
Supervised sentiment model trained on local review ratings.

Uses TF-IDF + Logistic Regression for a fast, domain-adapted classifier.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


class SupervisedSentimentAnalyzer:
    """Train and run a supervised sentiment classifier from local data."""

    def __init__(
        self,
        data_path: Optional[Path] = None,
        fallback_path: Optional[Path] = None,
        text_col: str = "text",
        rating_col: str = "rating",
    ):
        self.data_path = data_path
        self.fallback_path = fallback_path
        self.text_col = text_col
        self.rating_col = rating_col

        self._pipeline: Optional[Pipeline] = None
        self._classes: Optional[list[str]] = None
        self._trained_on: Optional[str] = None

    def predict(self, text: str) -> Dict[str, float | str | Dict[str, float]]:
        """Predict sentiment for a single text."""
        self._ensure_trained()
        cleaned = "" if pd.isna(text) else str(text)
        probabilities = self._pipeline.predict_proba([cleaned])[0]
        label = self._pipeline.predict([cleaned])[0]
        score = float(probabilities[self._classes.index(label)])
        # Expose class probabilities to explain the prediction.
        proba_map = {cls: round(float(prob), 4) for cls, prob in zip(self._classes, probabilities)}
        return {"label": label, "score": round(score, 4), "probabilities": proba_map}

    def predict_batch(
        self,
        df: pd.DataFrame,
        text_col: str = "text",
        prefix: str = "ml_",
    ) -> pd.DataFrame:
        """Add supervised sentiment columns to a DataFrame."""
        if text_col not in df.columns:
            raise ValueError(f"Column '{text_col}' not found.")

        self._ensure_trained()
        output = df.copy().reset_index(drop=True)
        texts = output[text_col].fillna("").astype(str).tolist()
        probabilities = self._pipeline.predict_proba(texts)
        labels = self._pipeline.predict(texts)

        result_df = pd.DataFrame(probabilities, columns=[f"{prefix}{cls.lower()}" for cls in self._classes])
        result_df[f"{prefix}label"] = labels
        max_scores = probabilities.max(axis=1)
        result_df[f"{prefix}score"] = max_scores.round(4)
        return pd.concat([output, result_df], axis=1)

    @property
    def is_trained(self) -> bool:
        return self._pipeline is not None

    @property
    def trained_on(self) -> Optional[str]:
        return self._trained_on

    def _ensure_trained(self) -> None:
        if self._pipeline is not None:
            return
        df = self._load_training_data()
        train_df = self._prepare_training_frame(df)
        if train_df.empty:
            raise ValueError("Training data has no usable labeled rows.")

        self._pipeline = Pipeline(
            steps=[
                (
                    "tfidf",
                    TfidfVectorizer(
                        max_features=6000,
                        ngram_range=(1, 2),
                        min_df=2,
                        max_df=0.95,
                    ),
                ),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=400,
                        class_weight="balanced",
                        multi_class="auto",
                    ),
                ),
            ]
        )

        X = train_df[self.text_col].tolist()
        y = train_df["label"].tolist()
        self._pipeline.fit(X, y)
        self._classes = list(self._pipeline.named_steps["clf"].classes_)

    def _load_training_data(self) -> pd.DataFrame:
        candidates = [self.data_path, self.fallback_path]
        for path in candidates:
            if path and Path(path).exists():
                self._trained_on = str(path)
                return pd.read_csv(path)
        raise ValueError("No training data file found for supervised sentiment.")

    def _prepare_training_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.text_col not in df.columns or self.rating_col not in df.columns:
            raise ValueError("Training data must include text and rating columns.")

        output = df[[self.text_col, self.rating_col]].copy()
        output[self.text_col] = output[self.text_col].fillna("").astype(str).str.strip()
        output[self.rating_col] = pd.to_numeric(output[self.rating_col], errors="coerce")
        output = output[(output[self.text_col].str.len() > 0) & output[self.rating_col].notna()].copy()
        output["label"] = output[self.rating_col].apply(self._label_from_rating)
        output = output[output["label"].notna()].reset_index(drop=True)
        return output

    @staticmethod
    def _label_from_rating(rating: float) -> Optional[str]:
        # Map star ratings into three sentiment classes.
        if rating >= 4:
            return "Positive"
        if rating <= 2:
            return "Negative"
        return "Neutral"
