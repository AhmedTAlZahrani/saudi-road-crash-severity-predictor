"""Smoke tests for the crash severity training pipeline.

Verifies that SMOTE pipelines can be constructed, fitted, and
cross-validated on a small synthetic dataset without crashing.
Real model quality is not asserted here.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

from src.train import train_and_compare, train_final_model, DEFAULT_MODELS


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def synthetic_Xy():
    """Create a small balanced dataset with 4 severity classes.

    Returns:
        Tuple of (X array, y array) with 200 samples and 10 features.
    """
    rng = np.random.default_rng(99)
    n_per_class = 50
    n_features = 10
    n = n_per_class * 4

    X = rng.standard_normal((n, n_features))
    y = np.repeat([0, 1, 2, 3], n_per_class)

    # Shuffle so stratified CV works correctly
    idx = rng.permutation(n)
    return X[idx], y[idx]


# ---------------------------------------------------------------------------
# DEFAULT_MODELS sanity
# ---------------------------------------------------------------------------

class TestDefaultModels:
    """Basic checks on the model registry."""

    def test_default_models_not_empty(self):
        """At least one model should be registered."""
        assert len(DEFAULT_MODELS) > 0

    @pytest.mark.parametrize("name", list(DEFAULT_MODELS.keys()))
    def test_each_model_has_fit_predict(self, name):
        """Every registered estimator must expose fit and predict."""
        model = DEFAULT_MODELS[name]
        assert hasattr(model, "fit")
        assert hasattr(model, "predict")


# ---------------------------------------------------------------------------
# train_and_compare smoke tests
# ---------------------------------------------------------------------------

class TestTrainAndCompare:
    """Smoke tests for the cross-validation comparison function."""

    def test_runs_with_single_fast_model(self, synthetic_Xy):
        """Pipeline should complete with a single lightweight model."""
        X, y = synthetic_Xy
        fast_models = {
            "LogReg": LogisticRegression(max_iter=200, random_state=42),
        }
        result = train_and_compare(X, y, models=fast_models, n_folds=2)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1
        assert "Model" in result.columns
        assert "f1" in result.columns

    def test_returns_sorted_by_f1(self, synthetic_Xy):
        """Results should be sorted by f1 descending."""
        X, y = synthetic_Xy
        two_models = {
            "LogReg": LogisticRegression(max_iter=200, random_state=42),
            "LogReg_C01": LogisticRegression(max_iter=200, C=0.01, random_state=42),
        }
        result = train_and_compare(X, y, models=two_models, n_folds=2)

        f1_values = result["f1"].tolist()
        assert f1_values == sorted(f1_values, reverse=True)

    def test_metric_columns_present(self, synthetic_Xy):
        """Expected metric columns must exist in the output."""
        X, y = synthetic_Xy
        fast_models = {
            "LogReg": LogisticRegression(max_iter=200, random_state=42),
        }
        result = train_and_compare(X, y, models=fast_models, n_folds=2)

        for col in ["accuracy", "precision", "recall", "f1"]:
            assert col in result.columns, f"Missing column: {col}"

    def test_metrics_are_between_zero_and_one(self, synthetic_Xy):
        """All metric values should be in [0, 1]."""
        X, y = synthetic_Xy
        fast_models = {
            "LogReg": LogisticRegression(max_iter=200, random_state=42),
        }
        result = train_and_compare(X, y, models=fast_models, n_folds=2)

        for col in ["accuracy", "precision", "recall", "f1"]:
            val = result[col].iloc[0]
            assert 0.0 <= val <= 1.0, f"{col}={val} out of [0,1]"


# ---------------------------------------------------------------------------
# train_final_model smoke tests
# ---------------------------------------------------------------------------

class TestTrainFinalModel:
    """Smoke tests for final model training and serialization."""

    def test_returns_fitted_pipeline(self, synthetic_Xy, tmp_path):
        """Trained pipeline should be able to predict."""
        X, y = synthetic_Xy
        pipeline = train_final_model(X, y, model_name="LogisticRegression",
                                     output_dir=str(tmp_path))

        assert isinstance(pipeline, ImbPipeline)
        preds = pipeline.predict(X)
        assert len(preds) == len(y)

    def test_saves_model_file(self, synthetic_Xy, tmp_path):
        """A .pkl model file should be created in the output directory."""
        X, y = synthetic_Xy
        train_final_model(X, y, model_name="LogisticRegression",
                          output_dir=str(tmp_path))

        pkl_files = list(tmp_path.glob("*.pkl"))
        assert len(pkl_files) == 1
        assert "logisticregression" in pkl_files[0].name.lower()

    def test_predictions_are_valid_classes(self, synthetic_Xy, tmp_path):
        """Predictions should only contain severity class labels 0-3."""
        X, y = synthetic_Xy
        pipeline = train_final_model(X, y, model_name="LogisticRegression",
                                     output_dir=str(tmp_path))

        preds = pipeline.predict(X)
        assert set(preds).issubset({0, 1, 2, 3})

    def test_invalid_model_name_raises(self, synthetic_Xy, tmp_path):
        """An unrecognized model name should raise a KeyError."""
        X, y = synthetic_Xy
        with pytest.raises(KeyError):
            train_final_model(X, y, model_name="NonExistentModel",
                              output_dir=str(tmp_path))
