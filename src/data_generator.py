"""Generate synthetic Saudi Arabia road crash data.

Creates realistic crash records calibrated to Saudi traffic statistics,
including Saudi-specific features like sandstorm visibility, extreme heat
conditions, Saher speed camera zones, and intercity highway patterns.
"""

import numpy as np
import pandas as pd
from pathlib import Path


# Saudi Arabia highway corridors with approximate km ranges
HIGHWAYS = {
    "E30_Jeddah_Riyadh": {"length_km": 950, "speed_limit": 120, "lanes": 3, "type": "intercity"},
    "E45_Riyadh_Dammam": {"length_km": 400, "speed_limit": 120, "lanes": 3, "type": "intercity"},
    "E11_Coastal": {"length_km": 700, "speed_limit": 120, "lanes": 2, "type": "intercity"},
    "E35_Madinah_Tabuk": {"length_km": 600, "speed_limit": 120, "lanes": 2, "type": "intercity"},
    "Riyadh_Ring_Road": {"length_km": 80, "speed_limit": 100, "lanes": 4, "type": "urban_highway"},
    "Jeddah_Corniche": {"length_km": 30, "speed_limit": 80, "lanes": 3, "type": "urban"},
    "Riyadh_King_Fahd_Rd": {"length_km": 25, "speed_limit": 80, "lanes": 3, "type": "urban"},
    "Dammam_Urban": {"length_km": 20, "speed_limit": 60, "lanes": 2, "type": "urban"},
    "Makkah_Madinah_Hwy": {"length_km": 420, "speed_limit": 120, "lanes": 3, "type": "intercity"},
    "Abha_Mountain_Rd": {"length_km": 80, "speed_limit": 80, "lanes": 2, "type": "mountain"},
}

VEHICLE_TYPES = ["sedan", "SUV", "pickup_truck", "semi_truck", "bus", "motorcycle"]
VEHICLE_WEIGHTS = [0.35, 0.25, 0.15, 0.10, 0.05, 0.10]

WEATHER_CONDITIONS = ["clear", "haze", "sandstorm", "rain", "fog"]
WEATHER_WEIGHTS_SUMMER = [0.65, 0.20, 0.10, 0.02, 0.03]
WEATHER_WEIGHTS_WINTER = [0.50, 0.15, 0.05, 0.20, 0.10]

SEVERITY_LEVELS = ["property_only", "minor_injury", "severe_injury", "fatal"]


def generate_crash_data(n_records=50000, seed=42):
    """Generate synthetic Saudi road crash records.

    Creates records with realistic Saudi-specific patterns including
    prayer time effects, extreme heat, sandstorms, Saher enforcement,
    and intercity highway fatality distributions.

    Args:
        n_records: Number of crash records to generate.
        seed: Random seed for reproducibility.

    Returns:
        DataFrame with crash records and severity labels.
    """
    rng = np.random.default_rng(seed)

    print(f"Generating {n_records:,} synthetic crash records...")

    # Temporal features
    dates = pd.date_range("2022-01-01", periods=n_records, freq=None)
    dates = pd.to_datetime(
        rng.choice(pd.date_range("2022-01-01", "2024-12-31"), size=n_records)
    )
    hours = rng.choice(24, size=n_records, p=_hour_distribution())

    month = dates.month
    day_of_week = dates.dayofweek
    is_weekend = (day_of_week >= 4).astype(int)  # Thu-Fri in Saudi Arabia
    is_ramadan = _is_ramadan_period(dates)
    is_hajj = _is_hajj_period(dates)

    # Road features
    highway_names = rng.choice(list(HIGHWAYS.keys()), size=n_records)
    road_info = pd.DataFrame([HIGHWAYS[h] for h in highway_names])
    speed_limit = road_info["speed_limit"].values
    num_lanes = road_info["lanes"].values
    road_type = road_info["type"].values

    # Vehicle features
    vehicle_type = rng.choice(VEHICLE_TYPES, size=n_records, p=VEHICLE_WEIGHTS)
    num_vehicles = rng.choice([1, 2, 3, 4, 5], size=n_records, p=[0.25, 0.45, 0.18, 0.08, 0.04])

    # Driver features
    driver_age = rng.normal(35, 12, size=n_records).clip(18, 75).astype(int)
    seatbelt = rng.choice([0, 1], size=n_records, p=[0.25, 0.75])
    is_saudi_national = rng.choice([0, 1], size=n_records, p=[0.35, 0.65])

    # Environmental features
    is_summer = ((month >= 5) & (month <= 9)).astype(int)
    weather = np.where(
        is_summer,
        rng.choice(WEATHER_CONDITIONS, size=n_records, p=WEATHER_WEIGHTS_SUMMER),
        rng.choice(WEATHER_CONDITIONS, size=n_records, p=WEATHER_WEIGHTS_WINTER),
    )
    temperature = np.where(
        is_summer,
        rng.normal(44, 5, size=n_records).clip(30, 55),
        rng.normal(22, 8, size=n_records).clip(5, 38),
    )

    visibility_km = _compute_visibility(weather, rng)

    # Saher (speed camera) zone
    in_saher_zone = rng.choice([0, 1], size=n_records, p=[0.6, 0.4])

    # Estimated speed at impact
    speed_factor = np.where(in_saher_zone, 0.85, 1.0)
    night_factor = np.where((hours >= 22) | (hours <= 5), 1.1, 1.0)
    estimated_speed = (speed_limit * speed_factor * night_factor *
                       rng.normal(1.0, 0.15, size=n_records)).clip(20, 200)

    # Rural/camel crossing risk (intercity highways at night)
    is_rural = (road_type == "intercity").astype(int)
    camel_crossing_risk = is_rural * ((hours >= 18) | (hours <= 6)).astype(int)

    # Prayer time flag (reduced traffic but higher speed)
    is_prayer_time = _is_prayer_time(hours, month)

    # Tire blowout risk (extreme heat + high speed)
    tire_blowout_risk = ((temperature > 45) & (estimated_speed > 100)).astype(int)

    # Build severity based on risk factors
    severity = _assign_severity(
        rng, n_records, estimated_speed, seatbelt, vehicle_type,
        weather, visibility_km, road_type, num_vehicles,
        camel_crossing_risk, tire_blowout_risk, driver_age, hours
    )

    df = pd.DataFrame({
        "date": dates,
        "hour": hours,
        "month": month,
        "day_of_week": day_of_week,
        "is_weekend": is_weekend,
        "is_ramadan": is_ramadan,
        "is_hajj": is_hajj,
        "highway": highway_names,
        "road_type": road_type,
        "speed_limit": speed_limit,
        "num_lanes": num_lanes,
        "vehicle_type": vehicle_type,
        "num_vehicles": num_vehicles,
        "driver_age": driver_age,
        "seatbelt": seatbelt,
        "is_saudi_national": is_saudi_national,
        "weather": weather,
        "temperature": np.round(temperature, 1),
        "visibility_km": np.round(visibility_km, 1),
        "in_saher_zone": in_saher_zone,
        "estimated_speed": np.round(estimated_speed, 1),
        "is_rural": is_rural,
        "camel_crossing_risk": camel_crossing_risk,
        "is_prayer_time": is_prayer_time,
        "tire_blowout_risk": tire_blowout_risk,
        "severity": severity,
    })

    _print_summary(df)
    return df


def _hour_distribution():
    """Return hourly crash probability distribution.

    Peaks during commute hours and late night (fatigue-related).
    Dips during early morning and prayer times.
    """
    probs = np.array([
        3, 2, 2, 1, 1, 2,    # 00-05: late night / early morning
        4, 7, 8, 6, 5, 5,    # 06-11: morning commute peak
        6, 5, 4, 5, 7, 8,    # 12-17: afternoon, post-Dhuhr dip, evening build
        8, 7, 6, 5, 4, 4,    # 18-23: evening peak then decline
    ], dtype=float)
    return probs / probs.sum()


def _is_ramadan_period(dates):
    """Approximate Ramadan periods (shifts each year by ~11 days)."""
    ramadan_starts = {
        2022: ("2022-04-02", "2022-05-01"),
        2023: ("2023-03-23", "2023-04-20"),
        2024: ("2024-03-11", "2024-04-09"),
    }
    flags = np.zeros(len(dates), dtype=int)
    for year, (start, end) in ramadan_starts.items():
        mask = (dates >= start) & (dates <= end)
        flags[mask] = 1
    return flags


def _is_hajj_period(dates):
    """Approximate Hajj periods."""
    hajj_periods = {
        2022: ("2022-07-07", "2022-07-12"),
        2023: ("2023-06-26", "2023-07-01"),
        2024: ("2024-06-14", "2024-06-19"),
    }
    flags = np.zeros(len(dates), dtype=int)
    for year, (start, end) in hajj_periods.items():
        mask = (dates >= start) & (dates <= end)
        flags[mask] = 1
    return flags


def _is_prayer_time(hours, months):
    """Flag approximate prayer times (varies by month/location).

    Simplified: Fajr ~4-5, Dhuhr ~12-13, Asr ~15-16, Maghrib ~18-19, Isha ~20-21.
    """
    prayer_hours = {4, 5, 12, 13, 15, 16, 18, 19, 20, 21}
    return np.array([1 if h in prayer_hours else 0 for h in hours])


def _compute_visibility(weather, rng):
    """Compute visibility based on weather conditions."""
    visibility = np.full(len(weather), 10.0)
    visibility[weather == "haze"] = rng.uniform(3, 7, size=(weather == "haze").sum())
    visibility[weather == "sandstorm"] = rng.uniform(0.1, 2, size=(weather == "sandstorm").sum())
    visibility[weather == "fog"] = rng.uniform(0.3, 3, size=(weather == "fog").sum())
    visibility[weather == "rain"] = rng.uniform(2, 8, size=(weather == "rain").sum())
    return visibility


def _assign_severity(rng, n, speed, seatbelt, vehicle_type, weather,
                     visibility, road_type, num_vehicles, camel_risk,
                     tire_risk, age, hours):
    """Assign crash severity based on weighted risk factors.

    Calibrated to Saudi statistics: ~65% property-only, 20% minor,
    10% severe, 5% fatal.
    """
    # Base risk score
    risk = np.zeros(n, dtype=float)

    # Speed is the dominant factor
    risk += (speed - 60) / 100 * 2.0

    # No seatbelt dramatically increases severity
    risk += (1 - seatbelt) * 1.5

    # Heavy vehicles cause worse crashes
    heavy = np.isin(vehicle_type, ["semi_truck", "bus"])
    risk += heavy * 1.0

    # Motorcycles are vulnerable
    risk += (vehicle_type == "motorcycle") * 2.0

    # Poor visibility
    risk += (1 / (visibility + 0.5)) * 0.8

    # Multi-vehicle crashes are worse
    risk += (num_vehicles - 1) * 0.5

    # Rural + night (camel crossing, no lighting)
    risk += camel_risk * 1.2

    # Tire blowout at high speed
    risk += tire_risk * 1.5

    # Young and elderly drivers
    risk += ((age < 25) | (age > 65)) * 0.5

    # Night driving (fatigue)
    risk += ((hours >= 0) & (hours <= 5)) * 0.8

    # Sandstorm pileups
    risk += (weather == "sandstorm") * 1.5

    # Mountain roads
    risk += (road_type == "mountain") * 0.7

    # Add noise
    risk += rng.normal(0, 0.8, size=n)

    # Map risk to severity using thresholds calibrated to target distribution
    severity = np.full(n, "property_only", dtype=object)
    severity[risk > 2.5] = "minor_injury"
    severity[risk > 4.0] = "severe_injury"
    severity[risk > 5.5] = "fatal"

    return severity


def _print_summary(df):
    """Print generation summary statistics."""
    print(f"\nGenerated {len(df):,} crash records")
    print(f"\nSeverity distribution:")
    dist = df["severity"].value_counts(normalize=True).sort_index()
    for level, pct in dist.items():
        print(f"  {level:20s}: {pct:6.1%}")
    print(f"\nRoad type distribution:")
    for rtype, count in df["road_type"].value_counts().items():
        print(f"  {rtype:20s}: {count:,}")
    print(f"\nDate range: {df['date'].min().date()} to {df['date'].max().date()}")


if __name__ == "__main__":
    output_dir = Path("data")
    output_dir.mkdir(parents=True, exist_ok=True)

    df = generate_crash_data(n_records=50000)
    df.to_csv(output_dir / "saudi_crashes.csv", index=False)
    print(f"\nSaved to {output_dir / 'saudi_crashes.csv'}")
