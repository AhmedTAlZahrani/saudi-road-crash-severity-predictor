"""FastAPI prediction server for crash severity classification."""

from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.transforms import build_crash_features

SEVERITY_LABELS = ["property_only", "minor_injury", "severe_injury", "fatal"]


class ModelRegistry:
    """Manages loading and access to serialized model artifacts.

    Provides a single point of control for model and preprocessor
    lifecycle within the API process.
    """

    def __init__(self, models_dir="models"):
        self.models_dir = Path(models_dir)
        self.model = None
        self.preprocessor = None

    def load(self, model_name="xgboost", preprocessor_name="preprocessor"):
        """Load model and optional preprocessor from disk.

        Tries ``{model_name}.pkl`` first, then falls back to ``best_model.pkl``
        if present, so deployments can override the default-trained model
        without touching the API code.

        Args:
            model_name: Filename stem for the model artifact (default
                ``xgboost`` since this is the best performer reported in
                the README).
            preprocessor_name: Filename stem for the preprocessor artifact.
        """
        for stem in (model_name, "best_model"):
            model_path = self.models_dir / f"{stem}.pkl"
            if model_path.exists():
                self.model = joblib.load(model_path)
                break

        prep_path = self.models_dir / f"{preprocessor_name}.pkl"
        if prep_path.exists():
            self.preprocessor = joblib.load(prep_path)

    @property
    def is_ready(self):
        return self.model is not None


registry = ModelRegistry()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model artifacts at startup. Newer FastAPI replacement for on_event."""
    registry.load()
    yield


app = FastAPI(
    title="Saudi Road Crash Severity Prediction API",
    description="Predict crash severity (property_only, minor, severe, fatal) with risk scoring.",
    version="1.0.0",
    lifespan=lifespan,
)


class CrashFeatures(BaseModel):
    hour: int = 14
    month: int = 7
    day_of_week: int = 2
    is_weekend: int = 0
    is_ramadan: int = 0
    is_hajj: int = 0
    highway: str = "E45_Riyadh_Dammam"
    road_type: str = "intercity"
    speed_limit: int = 120
    num_lanes: int = 3
    vehicle_type: str = "sedan"
    num_vehicles: int = 2
    driver_age: int = 28
    seatbelt: int = 1
    is_saudi_national: int = 1
    weather: str = "clear"
    temperature: float = 42.0
    visibility_km: float = 10.0
    in_saher_zone: int = 0
    estimated_speed: float = 130.0
    is_rural: int = 1
    camel_crossing_risk: int = 0
    is_prayer_time: int = 0
    tire_blowout_risk: int = 0


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": registry.is_ready}


@app.get("/model-info")
def model_info():
    if not registry.is_ready:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {
        "model_type": type(registry.model).__name__,
        "severity_levels": SEVERITY_LABELS,
    }


@app.post("/predict")
def predict(features: CrashFeatures):
    """Predict crash severity for given conditions."""
    if not registry.is_ready:
        raise HTTPException(status_code=503, detail="Model not loaded")

    df = pd.DataFrame([features.model_dump()])
    df = build_crash_features(df)

    if registry.preprocessor is not None:
        df = registry.preprocessor.transform(df)

    prediction = int(registry.model.predict(df)[0])
    severity = SEVERITY_LABELS[prediction]

    # Get probabilities if available
    proba = {}
    if hasattr(registry.model, "predict_proba"):
        probas = registry.model.predict_proba(df)[0]
        proba = {SEVERITY_LABELS[i]: round(float(p), 4) for i, p in enumerate(probas)}

    return {
        "severity_prediction": severity,
        "severity_code": prediction,
        "probabilities": proba,
        "risk_level": "CRITICAL" if prediction == 3 else (
            "HIGH" if prediction == 2 else (
                "MEDIUM" if prediction == 1 else "LOW"
            )
        ),
    }
