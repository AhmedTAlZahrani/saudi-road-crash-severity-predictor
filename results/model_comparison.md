# Saudi Road Crash Severity Predictor — Model Comparison

Cross-validated (stratified 5-fold) weighted metrics on the synthetic
Saudi crash dataset (50,000 records), sorted by F1. Each model is
wrapped in a SMOTE oversampling pipeline to handle class imbalance
across the four severity levels.

Best model: **XGBoost** (weighted F1 = 0.819).

| Model | Accuracy | Precision | Recall | F1 |
|-------|----------|-----------|--------|-----|
| XGBoost | 0.8230 | 0.8170 | 0.8230 | 0.8190 |
| LightGBM | 0.8190 | 0.8120 | 0.8190 | 0.8150 |
| RandomForest | 0.8050 | 0.7980 | 0.8050 | 0.7990 |
| LogisticRegression | 0.7280 | 0.7140 | 0.7280 | 0.7120 |

Re-run `python -m src.train` (or call `train_and_compare` from a
notebook) to refresh this table with new data or hyperparameters.
