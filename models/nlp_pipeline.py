"""
nlp_pipeline.py
---------------
End-to-end preprocessing, vectorization, sentiment, topic and NER pipeline.
"""

from typing import Dict, Optional

import pandas as pd

from models.ner import NamedEntityRecognizer
from models.sentiment_vader import VADERSentimentAnalyzer
from models.topic_model import LDATopicModel
from preprocessing.cleaner import TextCleaner
from preprocessing.vectorizer import TextVectorizer
from utils.metrics import validate_against_ratings


class NLPPipeline:
    """Run the complete Phase 2 + Phase 3 workflow on a reviews DataFrame."""

    def __init__(
        self,
        n_topics: int = 5,
        topic_passes: int = 10,
        embedding_dim: int = 50,
    ):
        self.cleaner = TextCleaner()
        self.vectorizer = TextVectorizer(min_df=1, embedding_dim=embedding_dim)
        self.sentiment = VADERSentimentAnalyzer()
        self.ner = NamedEntityRecognizer()
        self.topic_model = LDATopicModel(n_topics=n_topics, passes=topic_passes, min_df=1)

        self.preprocessing_report: Optional[Dict] = None
        self.validation_report: Optional[Dict] = None
        self.tfidf_shape = None
        self.embedding_shape = None

    def run(
        self,
        df: pd.DataFrame,
        text_col: str = "text",
        include_topics: bool = True,
        include_validation: bool = True,
    ) -> pd.DataFrame:
        """Return a DataFrame enriched with all NLP results."""
        # Cleaning, sentiment, and NER happen before vectorization/topic modeling.
        output, self.preprocessing_report = self.cleaner.prepare_dataframe(
            df,
            text_col=text_col,
            drop_duplicates=True,
        )
        output = self.sentiment.predict_batch(output, text_col=text_col)
        output = self.ner.predict_batch(output, text_col=text_col)

        output, tfidf_matrix, embeddings = self.vectorizer.build_feature_frame(
            output,
            text_col="token_text",
            include_embeddings=True,
        )
        self.tfidf_shape = tfidf_matrix.shape
        self.embedding_shape = None if embeddings is None else embeddings.shape

        if include_topics and len(output) >= 2:
            token_lists = output["tokens"].tolist()
            self.topic_model.fit(token_lists)
            output = self.topic_model.assign_topics(output, token_col="tokens")

        if include_validation and "rating" in output.columns:
            self.validation_report = validate_against_ratings(output)

        return output

    def report(self) -> Dict:
        """Return compact metadata for the latest pipeline run."""
        return {
            "preprocessing": self.preprocessing_report,
            "validation": self.validation_report,
            "tfidf_shape": self.tfidf_shape,
            "embedding_shape": self.embedding_shape,
            "ner_uses_spacy_model": self.ner.uses_spacy_model,
            "topics_fitted": self.topic_model.is_fitted,
        }
