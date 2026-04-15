"""Feature transforms for crash severity prediction.

Standalone functions for building Saudi-specific derived features
and risk indices from raw crash data.
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

from .data_loader import get_feature_types


def build_crash_features(df):
    """Build the full feature set from raw crash data.

    Applies speed features, weather interactions, and all derived
    risk indices. Returns a preprocessed DataFrame ready for modeling.

    Args:
        df: Raw crash feature DataFrame.

    Returns:
        DataFrame with all engineered features added.

    Raises:
        ValueError: If required columns are missing from the input.
    """
    required_cols = {"estimated_speed", "speed_limit"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"Input DataFrame is missing required columns: {sorted(missing)}. "
            f"Available columns: {sorted(df.columns.tolist())}"
        )

    if df.empty:
        raise ValueError("Input DataFrame is empty — cannot build crash features")

    df = df.copy()
    df = add_speed_features(df)
    df = add_weather_interactions(df)

    # Night driving flag (10 PM - 5 AM)
    if "hour" in df.columns:
        df["is_night"] = ((df["hour"] >= 22) | (df["hour"] <= 5)).astype(int)
        df["is_rush_hour"] = df["hour"].isin([7, 8, 9, 16, 17, 18]).astype(int)

    # Young/elderly driver risk
    if "driver_age" in df.columns:
        df["young_driver"] = (df["driver_age"] < 25).astype(int)
        df["elderly_driver"] = (df["driver_age"] > 65).astype(int)

    # Compound risk: high speed + no seatbelt
    if {"estimated_speed", "seatbelt"}.issubset(df.columns):
        df["speed_no_seatbelt"] = df["estimated_speed"] * (1 - df["seatbelt"])

    # Extreme heat flag (>45C)
    if "temperature" in df.columns:
        df["extreme_heat"] = (df["temperature"] > 45).astype(int)

    # Encode categoricals and scale numerics
    preprocessor = build_preprocessor(df)
    transformed = preprocessor.transform(df)

    # Build output DataFrame with feature names
    num_cols, cat_cols = get_feature_types(df)
    ohe = preprocessor.named_transformers_["cat"]
    cat_feature_names = ohe.get_feature_names_out(cat_cols).tolist()
    all_names = num_cols + cat_feature_names

    return pd.DataFrame(transformed, columns=all_names, index=df.index)


def add_speed_features(df):
    """Derive speed-related risk features.

    Args:
        df: DataFrame with speed columns.

    Returns:
        DataFrame with speed_over_limit and speed_ratio columns added.
    """
    df = df.copy()
    if {"estimated_speed", "speed_limit"}.issubset(df.columns):
        df["speed_over_limit"] = df["estimated_speed"] - df["speed_limit"]
        df["speed_ratio"] = df["estimated_speed"] / df["speed_limit"]
    return df


def add_weather_interactions(df):
    """Create weather and visibility interaction features.

    Args:
        df: DataFrame with weather-related columns.

    Returns:
        DataFrame with weather interaction columns added.
    """
    df = df.copy()

    # Compound risk: high speed + poor visibility
    if {"estimated_speed", "visibility_km"}.issubset(df.columns):
        df["speed_visibility_risk"] = df["estimated_speed"] / (df["visibility_km"] + 0.5)

    # Sandstorm + highway interaction
    if {"weather", "road_type"}.issubset(df.columns):
        df["sandstorm_highway"] = (
            (df["weather"] == "sandstorm") & (df["road_type"] == "intercity")
        ).astype(int)

    return df


def build_preprocessor(X):
    """Fit a ColumnTransformer for numeric scaling and categorical encoding.

    Args:
        X: Feature DataFrame (should already have derived features).

    Returns:
        Fitted ColumnTransformer instance.
    """
    numeric_cols, categorical_cols = get_feature_types(X)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(drop="first", sparse_output=False,
                                  handle_unknown="ignore"),
             categorical_cols),
        ],
        remainder="drop",
    )
    preprocessor.fit(X)
    return preprocessor
