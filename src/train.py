"""Multi-model training and comparison for crash severity classification.

Supports Logistic Regression, Random Forest, XGBoost, and LightGBM
with SMOTE oversampling for class imbalance handling.
"""

import logging
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

logger = logging.getLogger("crash_predictor")

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


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


def _write_results_summary(comparison, results_dir=RESULTS_DIR):
    """Persist the cross-validated comparison as CSV + markdown."""
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    csv_path = results_dir / "model_comparison.csv"
    md_path = results_dir / "model_comparison.md"

    comparison.to_csv(csv_path, index=False)

    best = comparison.iloc[0]
    lines = [
        "# Saudi Road Crash Severity Predictor — Model Comparison",
        "",
        "Cross-validated weighted metrics on the synthetic crash dataset, "
        "sorted by F1.",
        "",
        f"Best model: **{best['Model']}** "
        f"(weighted F1 = {best['f1']:.4f}).",
        "",
        "| Model | Accuracy | Precision | Recall | F1 |",
        "|-------|----------|-----------|--------|-----|",
    ]
    for _, r in comparison.iterrows():
        lines.append(
            f"| {r['Model']} | {r['accuracy']:.4f} | "
            f"{r['precision']:.4f} | {r['recall']:.4f} | {r['f1']:.4f} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return csv_path, md_path


def train_and_compare(X, y, models=None, n_folds=5, save_results=True):
    """Train multiple classifiers and return cross-validated comparison.

    Each model is wrapped in a SMOTE pipeline to handle class imbalance.
    Results are sorted by weighted F1 score.

    Args:
        X: Feature matrix (DataFrame or array).
        y: Target vector (0-3 severity levels).
        models: Dict of model name -> estimator. Uses DEFAULT_MODELS if None.
        n_folds: Number of stratified CV folds.
        save_results: If True, write the comparison to ``results/`` as CSV
            and markdown for later reference.

    Returns:
        DataFrame with cross-validation metrics per model, sorted by F1.
    """
    if models is None:
        models = DEFAULT_MODELS

    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    scoring = ["accuracy", "precision_weighted", "recall_weighted", "f1_weighted"]

    logger.info("Starting model comparison with %d models, %d folds", len(models), n_folds)

    rows = []
    for name, estimator in models.items():
        logger.info("Evaluating %s", name)
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
    logger.info("Best model: %s (weighted F1=%.4f)", best["Model"], best["f1"])
    print(f"\nBest model: {best['Model']} (weighted F1={best['f1']:.4f})")

    if save_results:
        csv_path, md_path = _write_results_summary(comparison)
        print(f"Saved comparison to {csv_path} and {md_path}")

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
    logger.info("Training final model: %s on %d samples", model_name, len(y))
    pipeline.fit(X, y)

    model_file = output_path / f"{model_name.lower()}.pkl"
    joblib.dump(pipeline, model_file)
    logger.info("Model saved to %s", model_file)
    print(f"Model saved to {model_file}")

    return pipeline
