"""Load and validate Saudi road crash data."""

import pandas as pd
import numpy as np


SEVERITY_ORDER = ["property_only", "minor_injury", "severe_injury", "fatal"]

EXPECTED_COLUMNS = [
    "hour", "month", "day_of_week", "is_weekend", "is_ramadan", "is_hajj",
    "highway", "road_type", "speed_limit", "num_lanes", "vehicle_type",
    "num_vehicles", "driver_age", "seatbelt", "is_saudi_national",
    "weather", "temperature", "visibility_km", "in_saher_zone",
    "estimated_speed", "is_rural", "camel_crossing_risk",
    "is_prayer_time", "tire_blowout_risk", "severity",
]


def load_crash_data(path="data/saudi_crashes.csv"):
    """Load and prepare the Saudi road crash dataset.

    Validates columns, encodes severity as ordered categories,
    and separates features from target.

    Args:
        path: Path to the CSV file.

    Returns:
        Tuple of (features DataFrame, target Series).
    """
    df = pd.read_csv(path)

    # Validate expected columns
    missing = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # Drop non-feature columns
    drop_cols = ["date"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # Encode severity as numeric target
    severity_map = {level: i for i, level in enumerate(SEVERITY_ORDER)}
    df["severity"] = df["severity"].map(severity_map)

    X = df.drop(columns=["severity"])
    y = df["severity"]

    print(f"Loaded {len(df):,} records | Severity distribution:")
    for level, code in severity_map.items():
        count = (y == code).sum()
        print(f"  {level:20s} ({code}): {count:,} ({count/len(y):.1%})")

    return X, y


def get_feature_types(X):
    """Identify numeric and categorical columns.

    Args:
        X: Feature DataFrame.

    Returns:
        Tuple of (numeric column names, categorical column names).
    """
    numeric = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical = X.select_dtypes(include=["object", "category"]).columns.tolist()
    return numeric, categorical

