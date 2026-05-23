"""
metrics.py
----------
Model evaluation utilities:
- Accuracy, Precision, Recall, F1
- Confusion matrix
- Agreement between VADER and BERT
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
import plotly.figure_factory as ff
import plotly.graph_objects as go


def evaluate_sentiment_model(
    df: pd.DataFrame,
    pred_col: str,
    true_col: Optional[str] = None,
) -> Dict:
    """
    Compute evaluation metrics.

    If true_col provided → supervised metrics (accuracy, F1...).
    Otherwise → descriptive statistics only.
    """
    if true_col and true_col in df.columns:
        return _supervised_metrics(df, pred_col, true_col)
    else:
        return _descriptive_metrics(df, pred_col)


def _supervised_metrics(df: pd.DataFrame, pred_col: str, true_col: str) -> Dict:
    from sklearn.metrics import (
        accuracy_score, f1_score, precision_score,
        recall_score, classification_report, confusion_matrix,
    )

    y_true = df[true_col].astype(str)
    y_pred = df[pred_col].astype(str)

    labels = sorted(y_true.unique().tolist())

    return {
        "accuracy":  round(accuracy_score(y_true, y_pred), 4),
        "f1_macro":  round(f1_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "precision": round(precision_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "recall":    round(recall_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "report":    classification_report(y_true, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "labels":    labels,
    }


def _descriptive_metrics(df: pd.DataFrame, pred_col: str) -> Dict:
    counts = df[pred_col].value_counts()
    total = len(df)
    return {
        "total": total,
        "distribution": {
            label: {
                "count":      int(counts.get(label, 0)),
                "percentage": round(counts.get(label, 0) / total * 100, 1),
            }
            for label in counts.index
        },
    }


def confusion_matrix_chart(
    confusion: list,
    labels: list,
) -> go.Figure:
    """Plotly heatmap confusion matrix."""
    z = np.array(confusion)
    text = [[str(v) for v in row] for row in z]

    fig = ff.create_annotated_heatmap(
        z=z,
        x=labels,
        y=labels,
        annotation_text=text,
        colorscale="Blues",
        showscale=True,
    )
    fig.update_layout(
        title="Confusion Matrix",
        xaxis_title="Predicted",
        yaxis_title="Actual",
        height=380,
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def model_comparison_chart(
    df: pd.DataFrame,
    vader_col: str = "vader_label",
    bert_col: str = "bert_label",
) -> go.Figure:
    """Bar chart comparing VADER vs BERT label distribution."""
    labels = ["Positive", "Negative", "Neutral"]

    vader_counts = df[vader_col].value_counts() if vader_col in df.columns else {}
    bert_counts  = df[bert_col].value_counts()  if bert_col  in df.columns else {}

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="VADER",
        x=labels,
        y=[int(vader_counts.get(l, 0)) for l in labels],
        marker_color="#6366f1",
    ))
    if bert_col in df.columns:
        fig.add_trace(go.Bar(
            name="DistilBERT",
            x=labels,
            y=[int(bert_counts.get(l, 0)) for l in labels],
            marker_color="#06b6d4",
        ))

    fig.update_layout(
        title="VADER vs DistilBERT Comparison",
        barmode="group",
        height=350,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def agreement_rate(df: pd.DataFrame, col1: str, col2: str) -> float:
    """Percentage of rows where two models agree."""
    if col1 not in df.columns or col2 not in df.columns:
        return 0.0
    return round((df[col1] == df[col2]).mean() * 100, 1)


def rating_to_sentiment(
    df: pd.DataFrame,
    rating_col: str = "rating",
    output_col: str = "rating_sentiment",
) -> pd.DataFrame:
    """
    Convert star ratings into sentiment labels for validation.

    4-5 stars -> Positive, 3 stars -> Neutral, 1-2 stars -> Negative.
    """
    if rating_col not in df.columns:
        raise ValueError(f"Column '{rating_col}' not found.")

    output = df.copy()
    rating = pd.to_numeric(output[rating_col], errors="coerce")
    output[output_col] = np.select(
        [rating >= 4, rating <= 2, rating == 3],
        ["Positive", "Negative", "Neutral"],
        default="Unknown",
    )
    return output


def validate_against_ratings(
    df: pd.DataFrame,
    pred_col: str = "vader_label",
    rating_col: str = "rating",
) -> Dict:
    """Evaluate a sentiment model against star ratings when available."""
    labeled = rating_to_sentiment(df, rating_col=rating_col)
    labeled = labeled[labeled["rating_sentiment"] != "Unknown"].copy()
    if labeled.empty:
        return {"error": "No valid ratings available for validation."}
    return evaluate_sentiment_model(labeled, pred_col=pred_col, true_col="rating_sentiment")
