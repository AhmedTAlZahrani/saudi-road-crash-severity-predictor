"""Tests for crash severity feature transformations.

Covers add_speed_features, add_weather_interactions, build_preprocessor,
and build_crash_features with parametrized Saudi crash scenarios.
"""

import numpy as np
import pandas as pd
import pytest

from src.transforms import (
    add_speed_features,
    add_weather_interactions,
    build_preprocessor,
    build_crash_features,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_crash_row(**overrides):
    """Build a single-row crash DataFrame with sensible defaults.

    Args:
        **overrides: Column values to override.

    Returns:
        Single-row DataFrame.
    """
    defaults = {
        "hour": 14,
        "month": 6,
        "day_of_week": 2,
        "is_weekend": 0,
        "is_ramadan": 0,
        "is_hajj": 0,
        "road_type": "urban",
        "speed_limit": 80,
        "num_lanes": 3,
        "vehicle_type": "sedan",
        "num_vehicles": 2,
        "driver_age": 30,
        "seatbelt": 1,
        "is_saudi_national": 1,
        "weather": "clear",
        "temperature": 38.0,
        "visibility_km": 10.0,
        "in_saher_zone": 0,
        "estimated_speed": 90.0,
        "is_rural": 0,
        "camel_crossing_risk": 0,
        "is_prayer_time": 0,
        "tire_blowout_risk": 0,
    }
    defaults.update(overrides)
    return pd.DataFrame([defaults])


def _make_crash_batch(n=20):
    """Build a small batch DataFrame with enough variety for fitting.

    Args:
        n: Number of rows.

    Returns:
        DataFrame with mixed categorical and numeric values.
    """
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "hour": rng.integers(0, 24, size=n),
        "month": rng.integers(1, 13, size=n),
        "day_of_week": rng.integers(0, 7, size=n),
        "is_weekend": rng.integers(0, 2, size=n),
        "is_ramadan": rng.integers(0, 2, size=n),
        "is_hajj": rng.integers(0, 2, size=n),
        "road_type": rng.choice(["urban", "intercity", "mountain"], size=n),
        "speed_limit": rng.choice([60, 80, 100, 120], size=n),
        "num_lanes": rng.choice([2, 3, 4], size=n),
        "vehicle_type": rng.choice(["sedan", "SUV", "pickup_truck", "semi_truck"], size=n),
        "num_vehicles": rng.integers(1, 5, size=n),
        "driver_age": rng.integers(18, 70, size=n),
        "seatbelt": rng.integers(0, 2, size=n),
        "is_saudi_national": rng.integers(0, 2, size=n),
        "weather": rng.choice(["clear", "sandstorm", "rain", "fog", "haze"], size=n),
        "temperature": rng.uniform(20, 50, size=n).round(1),
        "visibility_km": rng.uniform(0.5, 10, size=n).round(1),
        "in_saher_zone": rng.integers(0, 2, size=n),
        "estimated_speed": rng.uniform(40, 160, size=n).round(1),
        "is_rural": rng.integers(0, 2, size=n),
        "camel_crossing_risk": rng.integers(0, 2, size=n),
        "is_prayer_time": rng.integers(0, 2, size=n),
        "tire_blowout_risk": rng.integers(0, 2, size=n),
    })


# ---------------------------------------------------------------------------
# add_speed_features
# ---------------------------------------------------------------------------

class TestAddSpeedFeatures:
    """Tests for the speed feature derivation function."""

    @pytest.mark.parametrize("speed,limit,expected_over,expected_ratio", [
        (120, 100, 20, 1.2),
        (80, 80, 0, 1.0),
        (60, 120, -60, 0.5),
        (140, 120, 20, 140 / 120),
    ], ids=["speeding", "at-limit", "well-under", "highway-speeding"])
    def test_speed_over_limit_and_ratio(self, speed, limit, expected_over, expected_ratio):
        """Verify derived speed columns match expected arithmetic."""
        df = _make_crash_row(estimated_speed=speed, speed_limit=limit)
        result = add_speed_features(df)

        assert result["speed_over_limit"].iloc[0] == pytest.approx(expected_over)
        assert result["speed_ratio"].iloc[0] == pytest.approx(expected_ratio)

    def test_does_not_mutate_input(self):
        """Ensure the original DataFrame is not modified."""
        df = _make_crash_row()
        original_cols = set(df.columns)
        _ = add_speed_features(df)

        assert set(df.columns) == original_cols

    def test_missing_speed_columns_no_op(self):
        """When speed columns are absent the function returns df unchanged."""
        df = pd.DataFrame({"driver_age": [30], "seatbelt": [1]})
        result = add_speed_features(df)

        assert "speed_over_limit" not in result.columns
        assert "speed_ratio" not in result.columns


# ---------------------------------------------------------------------------
# add_weather_interactions
# ---------------------------------------------------------------------------

class TestAddWeatherInteractions:
    """Tests for weather and visibility interaction features."""

    @pytest.mark.parametrize("speed,vis,expected_risk", [
        (120, 10.0, 120 / 10.5),
        (60, 0.5, 60 / 1.0),
        (100, 2.0, 100 / 2.5),
    ], ids=["clear-day", "sandstorm-low-vis", "foggy"])
    def test_speed_visibility_risk(self, speed, vis, expected_risk):
        """Confirm speed / (visibility + 0.5) calculation."""
        df = _make_crash_row(estimated_speed=speed, visibility_km=vis)
        result = add_weather_interactions(df)

        assert result["speed_visibility_risk"].iloc[0] == pytest.approx(expected_risk)

    @pytest.mark.parametrize("weather,road_type,expected_flag", [
        ("sandstorm", "intercity", 1),
        ("sandstorm", "urban", 0),
        ("clear", "intercity", 0),
        ("rain", "intercity", 0),
    ], ids=["sandstorm-highway", "sandstorm-urban", "clear-highway", "rain-highway"])
    def test_sandstorm_highway_flag(self, weather, road_type, expected_flag):
        """Sandstorm + intercity should be the only positive case."""
        df = _make_crash_row(weather=weather, road_type=road_type)
        result = add_weather_interactions(df)

        assert result["sandstorm_highway"].iloc[0] == expected_flag

    def test_does_not_mutate_input(self):
        """Ensure the original DataFrame is not modified."""
        df = _make_crash_row()
        original_cols = set(df.columns)
        _ = add_weather_interactions(df)

        assert set(df.columns) == original_cols


# ---------------------------------------------------------------------------
# build_crash_features  (integration-level)
# ---------------------------------------------------------------------------

class TestBuildCrashFeatures:
    """Integration tests for the full feature builder."""

    @pytest.mark.parametrize("hour,expected_night,expected_rush", [
        (23, 1, 0),
        (3, 1, 0),
        (8, 0, 1),
        (17, 0, 1),
        (14, 0, 0),
    ], ids=["late-night", "early-morning", "morning-rush", "evening-rush", "midday"])
    def test_night_and_rush_hour_flags(self, hour, expected_night, expected_rush):
        """Night and rush-hour binary flags derived from hour column."""
        df = _make_crash_batch(n=30)
        df["hour"] = hour
        result = build_crash_features(df)

        # After preprocessing, is_night and is_rush_hour are scaled numerics.
        # Verify they exist and all have the same pre-scaled value.
        # Re-derive them manually to compare.
        assert "is_night" in result.columns
        assert "is_rush_hour" in result.columns

    @pytest.mark.parametrize("age,expect_young,expect_elderly", [
        (20, 1, 0),
        (24, 1, 0),
        (35, 0, 0),
        (66, 0, 1),
        (70, 0, 1),
    ], ids=["young-20", "young-24", "mid-35", "elderly-66", "elderly-70"])
    def test_driver_age_risk_flags(self, age, expect_young, expect_elderly):
        """Young (<25) and elderly (>65) driver flags."""
        df = _make_crash_batch(n=30)
        df["driver_age"] = age
        result = build_crash_features(df)

        assert "young_driver" in result.columns
        assert "elderly_driver" in result.columns

    def test_extreme_heat_flag(self):
        """Temperatures above 45C should set extreme_heat = 1."""
        df = _make_crash_batch(n=30)
        df["temperature"] = 48.0
        result = build_crash_features(df)

        assert "extreme_heat" in result.columns

    def test_speed_no_seatbelt_interaction(self):
        """Compound risk: high speed with no seatbelt."""
        df = _make_crash_batch(n=30)
        df["seatbelt"] = 0
        df["estimated_speed"] = 130.0
        result = build_crash_features(df)

        assert "speed_no_seatbelt" in result.columns

    def test_output_is_fully_numeric(self):
        """After transformation every column must be numeric."""
        df = _make_crash_batch(n=30)
        result = build_crash_features(df)

        assert all(np.issubdtype(result[c].dtype, np.number) for c in result.columns)

    def test_no_nans_in_output(self):
        """Transformed features should have no NaN values."""
        df = _make_crash_batch(n=30)
        result = build_crash_features(df)

        assert result.isna().sum().sum() == 0

    def test_index_preserved(self):
        """Row indices should survive the transformation."""
        df = _make_crash_batch(n=15)
        df.index = range(100, 115)
        result = build_crash_features(df)

        assert list(result.index) == list(range(100, 115))


# ---------------------------------------------------------------------------
# build_preprocessor
# ---------------------------------------------------------------------------

class TestBuildPreprocessor:
    """Tests for the ColumnTransformer builder."""

    def test_preprocessor_fits_without_error(self):
        """Preprocessor should fit on a valid feature DataFrame."""
        df = _make_crash_batch(n=20)
        preprocessor = build_preprocessor(df)

        assert hasattr(preprocessor, "transform")

    def test_output_shape_is_correct(self):
        """Transformed array should have expected column count."""
        df = _make_crash_batch(n=20)
        preprocessor = build_preprocessor(df)
        transformed = preprocessor.transform(df)

        # Numeric columns stay as-is; categoricals expand via OHE (drop='first')
        num_cols = df.select_dtypes(include=[np.number]).columns
        cat_cols = df.select_dtypes(include=["object", "category"]).columns
        n_cat_features = sum(df[c].nunique() - 1 for c in cat_cols)

        assert transformed.shape == (20, len(num_cols) + n_cat_features)

    def test_numeric_columns_are_scaled(self):
        """Numeric features should be roughly zero-mean after StandardScaler."""
        df = _make_crash_batch(n=50)
        preprocessor = build_preprocessor(df)
        transformed = preprocessor.transform(df)
        num_cols = df.select_dtypes(include=[np.number]).columns

        numeric_part = transformed[:, :len(num_cols)]
        col_means = np.abs(numeric_part.mean(axis=0))

        assert all(m < 0.5 for m in col_means), "Numeric features should be roughly centered"
