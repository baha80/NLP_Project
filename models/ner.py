"""
ner.py
------
Named Entity Recognition for customer reviews.
"""

import re
from collections import Counter
from typing import Dict, List

import pandas as pd


class NamedEntityRecognizer:
    """
    Extract named entities using spaCy when available.

    If the English spaCy model is missing, a conservative regex fallback still
    extracts likely brand/product/location phrases so the pipeline remains
    usable in offline classroom environments.
    """

    FALLBACK_LABEL = "ENTITY"

    def __init__(self, model_name: str = "en_core_web_sm"):
        self.model_name = model_name
        self._nlp = None
        self._uses_spacy_model = False

    def _load(self):
        if self._nlp is not None:
            return

        try:
            import spacy

            self._nlp = spacy.load(self.model_name)
            self._uses_spacy_model = True
        except Exception:
            self._nlp = None
            self._uses_spacy_model = False

    def extract(self, text: str) -> List[Dict]:
        """Extract entities from a single text."""
        self._load()
        raw_text = "" if pd.isna(text) else str(text)

        if self._uses_spacy_model:
            # Use spaCy NER when the model is available.
            doc = self._nlp(raw_text)
            return [
                {
                    "text": ent.text,
                    "label": ent.label_,
                    "start": int(ent.start_char),
                    "end": int(ent.end_char),
                }
                for ent in doc.ents
            ]

        return self._fallback_extract(raw_text)

    def predict_batch(
        self,
        df: pd.DataFrame,
        text_col: str = "text",
        prefix: str = "ner_",
    ) -> pd.DataFrame:
        """Add entity columns to a DataFrame."""
        if text_col not in df.columns:
            raise ValueError(f"Column '{text_col}' not found.")

        output = df.copy()
        entities = output[text_col].apply(self.extract)
        output[f"{prefix}entities"] = entities
        output[f"{prefix}entity_count"] = entities.apply(len)
        output[f"{prefix}entity_text"] = entities.apply(
            lambda items: ", ".join(item["text"] for item in items)
        )
        return output

    def summarize_entities(
        self,
        df: pd.DataFrame,
        entity_col: str = "ner_entities",
        top_n: int = 20,
    ) -> pd.DataFrame:
        """Return the most frequent extracted entities."""
        if entity_col not in df.columns:
            raise ValueError(f"Column '{entity_col}' not found.")

        counts = Counter()
        labels = {}
        for entities in df[entity_col]:
            for ent in entities:
                key = ent["text"].strip()
                if key:
                    counts[key] += 1
                    labels[key] = ent.get("label", self.FALLBACK_LABEL)

        rows = [
            {"entity": entity, "label": labels.get(entity, ""), "count": count}
            for entity, count in counts.most_common(top_n)
        ]
        return pd.DataFrame(rows)

    @property
    def uses_spacy_model(self) -> bool:
        self._load()
        return self._uses_spacy_model

    def _fallback_extract(self, text: str) -> List[Dict]:
        # Regex fallback to keep the app usable without spaCy model downloads.
        pattern = re.compile(r"\b(?:[A-Z][a-zA-Z0-9&'-]+)(?:\s+[A-Z][a-zA-Z0-9&'-]+)*\b")
        ignored = {
            "I", "The", "This", "That", "It", "Very", "Great", "Bad", "Good",
            "Terrible", "Perfect", "Excellent", "Awful", "Customer",
        }

        entities = []
        for match in pattern.finditer(text):
            value = match.group(0).strip()
            if value in ignored or len(value) < 3:
                continue
            entities.append(
                {
                    "text": value,
                    "label": self.FALLBACK_LABEL,
                    "start": int(match.start()),
                    "end": int(match.end()),
                }
            )
        return entities
