"""Predict diabetes risk for a single patient using the trained pipeline.

Because the saved artifact is a full Pipeline (imputer + scaler + model), we do
NOT need to re-implement any preprocessing here -- we only have to reproduce the
one data-quality rule from training: a recorded value of 0 for Glucose, Blood
Pressure, Skin Thickness, Insulin or BMI is physiologically impossible and means
"not measured", so we convert it to missing and let the pipeline impute it.

Example:
    python src/predict.py
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
from joblib import load

from data import FEATURES, ZERO_AS_MISSING

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models",
    "diabetes_pipeline.joblib",
)


def load_model(model_path: str = MODEL_PATH):
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"No trained model at {model_path}. Run `python src/train.py` first."
        )
    return load(model_path)


def risk_band(probability: float) -> str:
    """Turn a probability into a human-readable screening band."""
    if probability < 0.30:
        return "Low"
    if probability < 0.60:
        return "Moderate"
    return "High"


def predict_patient(patient: dict, artifact=None) -> dict:
    """Predict risk for one patient.

    ``patient`` is a dict keyed by the 8 feature names. Missing/unknown values
    can be passed as 0 or omitted -- they are treated as "not measured".
    """
    artifact = artifact or load_model()
    pipeline = artifact["pipeline"]

    row = {feature: patient.get(feature, np.nan) for feature in FEATURES}
    X = pd.DataFrame([row], columns=FEATURES)
    # Apply the same "0 means missing" rule used during training.
    X[ZERO_AS_MISSING] = X[ZERO_AS_MISSING].replace(0, np.nan)

    probability = float(pipeline.predict_proba(X)[0, 1])
    return {
        "probability": probability,
        "prediction": int(probability >= 0.5),
        "risk_band": risk_band(probability),
        "model": artifact.get("model_name", "unknown"),
    }


if __name__ == "__main__":
    # Demo patient (values in the original dataset's units).
    example = {
        "Pregnancies": 6,
        "Glucose": 148,
        "BloodPressure": 72,
        "SkinThickness": 35,
        "Insulin": 0,  # unknown -> treated as missing
        "BMI": 33.6,
        "DiabetesPedigree": 0.627,
        "Age": 50,
    }
    result = predict_patient(example)
    print(f"Model         : {result['model']}")
    print(f"Risk probability: {result['probability']:.1%}")
    print(f"Risk band       : {result['risk_band']}")
    print(f"Predicted class : {'Diabetes' if result['prediction'] else 'No diabetes'}")
