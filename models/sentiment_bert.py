"""
Optional DistilBERT sentiment comparison.

The app keeps this model lazy because the first HuggingFace download can be
large. VADER remains the default production path.
"""

from __future__ import annotations

from typing import Dict, Optional

import pandas as pd


class BERTSentimentAnalyzer:
    """HuggingFace DistilBERT sentiment classifier."""

    MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"

    LABEL_MAP = {
        "POSITIVE": "Positive",
        "NEGATIVE": "Negative",
        "positive": "Positive",
        "negative": "Negative",
        "neutral": "Neutral",
        "LABEL_0": "Negative",
        "LABEL_1": "Neutral",
        "LABEL_2": "Positive",
    }

    def __init__(
        self,
        model_name: Optional[str] = None,
        use_gpu: bool = False,
        neutral_confidence_cutoff: float = 0.65,
    ):
        self.model_name = model_name or self.MODEL_NAME
        self.device = 0 if use_gpu else -1
        self.neutral_confidence_cutoff = neutral_confidence_cutoff
        self._pipeline = None

    def _load(self) -> None:
        if self._pipeline is not None:
            return

        from transformers import pipeline

        # Lazy-load to avoid large downloads unless the model is used.
        self._pipeline = pipeline(
            "sentiment-analysis",
            model=self.model_name,
            device=self.device,
            truncation=True,
            max_length=512,
        )

    def _normalize_result(self, raw: Dict) -> Dict[str, float | str]:
        raw_label = str(raw.get("label", "Neutral"))
        score = float(raw.get("score", 0.0))
        label = self.LABEL_MAP.get(raw_label, raw_label.capitalize())

        # DistilBERT SST-2 is binary; low confidence is treated as neutral.
        if score < self.neutral_confidence_cutoff:
            label = "Neutral"

        return {"label": label, "score": round(score, 4)}

    def predict(self, text: str) -> Dict[str, float | str]:
        self._load()
        raw = self._pipeline(str(text)[:2000])[0]
        return self._normalize_result(raw)

    def predict_batch(
        self,
        df: pd.DataFrame,
        text_col: str = "text",
        batch_size: int = 16,
        prefix: str = "bert_",
    ) -> pd.DataFrame:
        if text_col not in df.columns:
            raise ValueError(f"Column '{text_col}' not found.")

        self._load()
        output = df.copy().reset_index(drop=True)
        texts = output[text_col].fillna("").astype(str).str[:2000].tolist()
        raw_results = self._pipeline(texts, batch_size=batch_size)
        results = [self._normalize_result(item) for item in raw_results]
        result_df = pd.DataFrame(results).add_prefix(prefix)
        return pd.concat([output, result_df], axis=1)

    @property
    def is_loaded(self) -> bool:
        return self._pipeline is not None
