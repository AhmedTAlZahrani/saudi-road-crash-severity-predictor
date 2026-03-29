"""One-command demo: generates data, trains models, evaluates, and starts the API."""

import os
import sys
from pathlib import Path

def main():
    # Ensure data directory exists
    Path("data").mkdir(exist_ok=True)
    Path("models").mkdir(exist_ok=True)
    Path("output").mkdir(exist_ok=True)

    print("=" * 60)
    print("  Saudi Road Crash Severity Predictor — Full Demo")
    print("=" * 60)

    # Step 1: Generate synthetic crash data
    print("\n[1/4] Generating 5000 synthetic crash records...")
    from src.simulate_crashes import generate_crash_data
    df = generate_crash_data(n_records=5000)
    df.to_csv("data/saudi_crashes.csv", index=False)
    print(f"  Saved to data/saudi_crashes.csv ({len(df)} rows)")

    # Step 2: Load and engineer features
    print("\n[2/4] Loading data and building features...")
    from src.data_loader import load_crash_data
    from src.transforms import build_crash_features
    X, y = load_crash_data("data/saudi_crashes.csv")
    X_processed = build_crash_features(X)
    print(f"  Features: {X_processed.shape[1]} columns, {X_processed.shape[0]} samples")

    # Step 3: Train and compare models
    print("\n[3/4] Training 4 models with 3-fold CV (this takes ~30 seconds)...")
    from src.train import train_and_compare, train_final_model
    results = train_and_compare(X_processed, y, n_folds=3)
    print("\n  Model Comparison:")
    print(results.to_string(index=False))

    # Save best model
    best_name = results.iloc[0]["Model"]
    print(f"\n[4/4] Training final {best_name} model on full dataset...")
    train_final_model(X_processed, y, model_name=best_name)

    print("\n" + "=" * 60)
    print("  Demo complete!")
    print("=" * 60)
    print(f"\nTo start the prediction API:")
    print(f"  uvicorn api.main:app --host 0.0.0.0 --port 8000")
    print(f"\nThen try a prediction:")
    print(f'  curl -X POST http://localhost:8000/predict \\')
    print(f'    -H "Content-Type: application/json" \\')
    print(f'    -d \'{{"estimated_speed": 145, "seatbelt": 0, "weather": "sandstorm"}}\'')


if __name__ == "__main__":
    main()
