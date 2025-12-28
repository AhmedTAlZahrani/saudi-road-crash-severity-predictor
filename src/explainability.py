"""SHAP-based model explainability for crash severity predictions.

Provides global feature importance and local instance-level
explanations to understand what drives crash severity.
"""

import shap
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path


class CrashExplainer:
    """SHAP explainability for crash severity models.

    Generates global summary plots, feature importance rankings,
    and instance-level force explanations.
    """

    def __init__(self, model, feature_names, output_dir="output"):
        self.model = model
        self.feature_names = feature_names
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.explainer = None
        self.shap_values = None

    def compute_shap_values(self, X, sample_size=500):
        """Compute SHAP values for a sample of the data.

        Uses TreeExplainer for tree-based models, falls back to
        KernelExplainer for others.

        Args:
            X: Feature matrix.
            sample_size: Number of samples for SHAP computation.

        Returns:
            SHAP values array.
        """
        if sample_size and len(X) > sample_size:
            idx = np.random.choice(len(X), sample_size, replace=False)
            X_sample = X.iloc[idx] if hasattr(X, "iloc") else X[idx]
        else:
            X_sample = X

        # Extract the actual model from the SMOTE pipeline if needed
        actual_model = self.model
        if hasattr(actual_model, "named_steps"):
            actual_model = actual_model.named_steps.get("model", actual_model)

        try:
            self.explainer = shap.TreeExplainer(actual_model)
            print("Using TreeExplainer for SHAP values...")
        except Exception:
            background = shap.sample(X_sample, min(100, len(X_sample)))
            self.explainer = shap.KernelExplainer(actual_model.predict, background)
            print("Using KernelExplainer for SHAP values...")

        self.shap_values = self.explainer.shap_values(X_sample)
        print(f"SHAP values computed for {len(X_sample)} samples")
        return self.shap_values

    def get_feature_importance(self, class_idx=None):
        """Get mean absolute SHAP values as feature importance.

        Args:
            class_idx: If multi-class, which class to explain.
                       None returns average across all classes.

        Returns:
            DataFrame with feature importance rankings.
        """
        if self.shap_values is None:
            raise ValueError("Call compute_shap_values() first")

        if isinstance(self.shap_values, list):
            # Multi-class: average across classes or pick one
            if class_idx is not None:
                values = np.abs(self.shap_values[class_idx])
            else:
                values = np.mean([np.abs(sv) for sv in self.shap_values], axis=0)
        else:
            values = np.abs(self.shap_values)

        importance = pd.DataFrame({
            "feature": self.feature_names[:values.shape[1]],
            "mean_shap": values.mean(axis=0),
        }).sort_values("mean_shap", ascending=False)

        return importance

    def plot_global_importance(self, top_n=15):
        """Plot top feature importances as a horizontal bar chart.

        Args:
            top_n: Number of top features to display.
        """
        importance = self.get_feature_importance().head(top_n)

        fig = go.Figure(go.Bar(
            x=importance["mean_shap"].values[::-1],
            y=importance["feature"].values[::-1],
            orientation="h",
            marker_color="rgb(55, 83, 109)",
        ))

        fig.update_layout(
            title=f"Top {top_n} Features — Mean |SHAP| Value",
            xaxis_title="Mean |SHAP| Value",
            yaxis_title="Feature",
            template="plotly_dark",
            height=500,
            margin=dict(l=200),
        )

        fig.write_html(str(self.output_dir / "shap_importance.html"))
        print(f"SHAP importance plot saved to {self.output_dir / 'shap_importance.html'}")

    def explain_instance(self, X, idx=0):
        """Generate explanation for a single crash instance.

        Args:
            X: Feature matrix.
            idx: Index of the instance to explain.

        Returns:
            Dict with feature contributions for the instance.
        """
        if self.explainer is None:
            raise ValueError("Call compute_shap_values() first")

        instance = X.iloc[[idx]] if hasattr(X, "iloc") else X[[idx]]
        sv = self.explainer.shap_values(instance)

        if isinstance(sv, list):
            # Use the predicted class
            pred_class = self.model.predict(instance)[0]
            contributions = sv[pred_class][0]
        else:
            contributions = sv[0]

        feature_names = self.feature_names[:len(contributions)]
        result = dict(zip(feature_names, contributions))
        result = dict(sorted(result.items(), key=lambda x: abs(x[1]), reverse=True))

        return result
