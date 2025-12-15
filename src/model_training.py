"""Multi-model training and comparison for crash severity classification.

Supports Logistic Regression, Random Forest, XGBoost, and LightGBM
with SMOTE oversampling for class imbalance handling.
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline


DEFAULT_MODELS = {
    "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
    "RandomForest": RandomForestClassifier(n_estimators=200, max_depth=12,
                                            random_state=42, n_jobs=-1),
    "XGBoost": XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                              objective="multi:softprob", num_class=4,
                              eval_metric="mlogloss", random_state=42),
    "LightGBM": LGBMClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                                 num_class=4, objective="multiclass",
                                 random_state=42, verbose=-1),
    # model = CatBoostClassifier(iterations=200)  # tried, similar perf but slower
}


def train_and_compare(X, y, models=None, n_folds=5):
    """Train multiple classifiers and return cross-validated comparison.

    Each model is wrapped in a SMOTE pipeline to handle class imbalance.
    Results are sorted by weighted F1 score.

    Args:
        X: Feature matrix (DataFrame or array).
        y: Target vector (0-3 severity levels).
        models: Dict of model name -> estimator. Uses DEFAULT_MODELS if None.
        n_folds: Number of stratified CV folds.

    Returns:
        DataFrame with cross-validation metrics per model, sorted by F1.
    """
    if models is None:
        models = DEFAULT_MODELS

    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    scoring = ["accuracy", "precision_weighted", "recall_weighted", "f1_weighted"]

    rows = []
    for name, estimator in models.items():
        print(f"  Evaluating {name}...")

        pipeline = ImbPipeline([
            ("smote", SMOTE(random_state=42)),
            ("model", estimator),
        ])

        results = cross_validate(pipeline, X, y, cv=cv, scoring=scoring,
                                  return_train_score=False, n_jobs=-1)

        row = {"Model": name}
        for metric in scoring:
            key = metric.replace("_weighted", "")
            row[key] = round(results[f"test_{metric}"].mean(), 4)
        rows.append(row)

    comparison = pd.DataFrame(rows).sort_values("f1", ascending=False)
    best = comparison.iloc[0]
    print(f"\nBest model: {best['Model']} (weighted F1={best['f1']:.4f})")

    return comparison


def train_final_model(X, y, model_name="XGBoost", output_dir="models"):
    """Train a single model on the full dataset and save it.

    Args:
        X: Feature matrix.
        y: Target vector.
        model_name: Key from DEFAULT_MODELS.
        output_dir: Directory for saving the serialized model.

    Returns:
        Fitted pipeline instance.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    pipeline = ImbPipeline([
        ("smote", SMOTE(random_state=42)),
        ("model", DEFAULT_MODELS[model_name]),
    ])
    pipeline.fit(X, y)

    model_file = output_path / f"{model_name.lower()}.pkl"
    joblib.dump(pipeline, model_file)
    print(f"Model saved to {model_file}")

    return pipeline
