"""Evaluation metrics and visualizations for crash severity models."""

import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.figure_factory as ff
from pathlib import Path
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)


SEVERITY_LABELS = ["property_only", "minor_injury", "severe_injury", "fatal"]


def evaluate_model(model, X_test, y_test, output_dir="output"):
    """Run full evaluation on a trained model.

    Generates classification report, confusion matrix, and
    per-class performance metrics.

    Args:
        model: Trained model or pipeline.
        X_test: Test feature matrix.
        y_test: True labels.
        output_dir: Directory to save outputs.

    Returns:
        Dict with evaluation metrics.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision_weighted": round(precision_score(y_test, y_pred, average="weighted"), 4),
        "recall_weighted": round(recall_score(y_test, y_pred, average="weighted"), 4),
        "f1_weighted": round(f1_score(y_test, y_pred, average="weighted"), 4),
    }

    print("\n" + "=" * 60)
    print("CLASSIFICATION REPORT")
    print("=" * 60)
    print(classification_report(y_test, y_pred, target_names=SEVERITY_LABELS))

    # Save metrics
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {output_dir / 'metrics.json'}")

    # Generate confusion matrix
    plot_confusion_matrix(y_test, y_pred, output_dir)

    # Per-class metrics
    per_class = classification_report(y_test, y_pred, target_names=SEVERITY_LABELS,
                                       output_dict=True)
    per_class_df = pd.DataFrame(per_class).T
    per_class_df.to_csv(output_dir / "per_class_metrics.csv")

    return metrics


def plot_confusion_matrix(y_true, y_pred, output_dir="output"):
    """Generate and save an interactive confusion matrix.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        output_dir: Directory to save the plot.
    """
    output_dir = Path(output_dir)
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig = ff.create_annotated_heatmap(
        z=cm_normalized,
        x=SEVERITY_LABELS,
        y=SEVERITY_LABELS,
        annotation_text=[[f"{val:.0f}\n({pct:.1%})" for val, pct in zip(row_cm, row_pct)]
                          for row_cm, row_pct in zip(cm, cm_normalized)],
        colorscale="RdYlGn_r",
        showscale=True,
    )

    fig.update_layout(
        title="Crash Severity Prediction — Confusion Matrix",
        xaxis_title="Predicted Severity",
        yaxis_title="Actual Severity",
        template="plotly_dark",
        width=700,
        height=600,
    )

    fig.write_html(str(output_dir / "confusion_matrix.html"))
    print(f"Confusion matrix saved to {output_dir / 'confusion_matrix.html'}")


def plot_severity_distribution(y_true, y_pred, output_dir="output"):
    """Compare actual vs predicted severity distributions.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        output_dir: Directory to save the plot.
    """
    output_dir = Path(output_dir)

    actual_counts = pd.Series(y_true).value_counts().sort_index()
    pred_counts = pd.Series(y_pred).value_counts().sort_index()

    fig = go.Figure(data=[
        go.Bar(name="Actual", x=SEVERITY_LABELS, y=[actual_counts.get(i, 0) for i in range(4)]),
        go.Bar(name="Predicted", x=SEVERITY_LABELS, y=[pred_counts.get(i, 0) for i in range(4)]),
    ])

    fig.update_layout(
        title="Actual vs Predicted Severity Distribution",
        xaxis_title="Severity Level",
        yaxis_title="Count",
        barmode="group",
        template="plotly_dark",
    )

    fig.write_html(str(output_dir / "severity_distribution.html"))
    print(f"Distribution plot saved to {output_dir / 'severity_distribution.html'}")
