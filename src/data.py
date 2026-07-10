"""Dataset loading and cleaning for the Pima Indians Diabetes dataset.

The dataset is downloaded from a public mirror on first run and cached locally
under ``data/``. Physiologically impossible zero values (e.g. a Glucose or BMI
of 0) are treated as MISSING here, but they are *not* imputed in this module.
Imputation happens inside the scikit-learn Pipeline (see ``train.py``) so that
the fill values are learned from the training fold only -- this is what
prevents data leakage into the test set.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

# Public mirror of the UCI Pima Indians Diabetes dataset.
DATA_URL = (
    "https://raw.githubusercontent.com/jbrownlee/Datasets/master/"
    "pima-indians-diabetes.data.csv"
)

COLUMNS = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigree",
    "Age",
    "Outcome",
]

TARGET = "Outcome"

FEATURES = [c for c in COLUMNS if c != TARGET]

# Columns where a value of 0 is physiologically impossible and therefore
# actually means "not recorded". We convert these to NaN so the pipeline's
# imputer can fill them properly.
ZERO_AS_MISSING = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

# Where we cache the raw CSV so we do not re-download every run.
_CACHE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    "pima-indians-diabetes.csv",
)


def load_raw(cache_path: str = _CACHE_PATH) -> pd.DataFrame:
    """Load the raw dataset, downloading and caching it if necessary."""
    if os.path.exists(cache_path):
        return pd.read_csv(cache_path, names=COLUMNS)

    df = pd.read_csv(DATA_URL, names=COLUMNS)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    df.to_csv(cache_path, header=False, index=False)
    return df


def mark_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Replace impossible zeros with NaN in the affected feature columns."""
    df = df.copy()
    df[ZERO_AS_MISSING] = df[ZERO_AS_MISSING].replace(0, np.nan)
    return df


def load_dataset(cache_path: str = _CACHE_PATH) -> tuple[pd.DataFrame, pd.Series]:
    """Return (X, y) with impossible zeros marked as missing (not yet imputed)."""
    df = mark_missing(load_raw(cache_path))
    X = df[FEATURES].copy()
    y = df[TARGET].astype(int).copy()
    return X, y
