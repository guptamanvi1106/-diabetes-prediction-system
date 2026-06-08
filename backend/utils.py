"""
utils.py
--------
Shared helpers: model loading, input validation, logging setup.
"""

import os
import logging
import numpy as np
import pandas as pd
import joblib
from logging.handlers import RotatingFileHandler

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH  = os.path.join(BASE_DIR, "model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")
LOG_DIR     = os.path.join(BASE_DIR, "logs")
UPLOAD_DIR  = os.path.join(BASE_DIR, "uploads")

os.makedirs(LOG_DIR,    exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ── Feature definition ────────────────────────────────────────────────────────
FEATURE_NAMES = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]

FEATURE_RANGES = {
    "Pregnancies":             (0, 20),
    "Glucose":                 (0, 300),
    "BloodPressure":           (0, 200),
    "SkinThickness":           (0, 100),
    "Insulin":                 (0, 900),
    "BMI":                     (0, 70),
    "DiabetesPedigreeFunction":(0, 3),
    "Age":                     (1, 120),
}


# ── Logger ────────────────────────────────────────────────────────────────────
def get_logger(name: str = "diabetes_api") -> logging.Logger:
    """Return a logger that writes to both console and a rotating file."""
    logger = logging.getLogger(name)
    if logger.handlers:          # already configured
        return logger

    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter(
        "%(asctime)s  [%(levelname)s]  %(name)s – %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console
    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)

    # File  (5 MB × 3 backups)
    fh = RotatingFileHandler(
        os.path.join(LOG_DIR, "api.log"),
        maxBytes=5 * 1024 * 1024,
        backupCount=3,
    )
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)

    logger.addHandler(sh)
    logger.addHandler(fh)
    return logger


logger = get_logger()


# ── Model loading ─────────────────────────────────────────────────────────────
_model  = None
_scaler = None


def load_model():
    """Load (and cache) model + scaler from disk."""
    global _model, _scaler
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"model.pkl not found at {MODEL_PATH}. "
                "Please run train_model.py first."
            )
        _model  = joblib.load(MODEL_PATH)
        _scaler = joblib.load(SCALER_PATH)
        logger.info("model.pkl and scaler.pkl loaded successfully.")
    return _model, _scaler


# ── Input validation ──────────────────────────────────────────────────────────
def validate_single_input(data: dict) -> tuple[dict, list[str]]:
    """
    Validate and coerce a single prediction request.
    Returns (cleaned_dict, errors).
    """
    errors = []
    cleaned = {}

    for feature in FEATURE_NAMES:
        if feature not in data:
            errors.append(f"Missing required field: '{feature}'")
            continue
        try:
            val = float(data[feature])
        except (TypeError, ValueError):
            errors.append(f"Field '{feature}' must be numeric, got: {data[feature]!r}")
            continue

        lo, hi = FEATURE_RANGES[feature]
        if not (lo <= val <= hi):
            errors.append(
                f"Field '{feature}' value {val} is out of expected range [{lo}, {hi}]"
            )
            continue

        cleaned[feature] = val

    return cleaned, errors


def dict_to_array(data: dict) -> np.ndarray:
    """Convert validated dict → (1, 8) numpy array in feature order."""
    return np.array([[data[f] for f in FEATURE_NAMES]])


def predict_single(data: dict) -> dict:
    """
    Run a single prediction.
    Returns dict with 'prediction' label and 'confidence' (0-1).
    """
    model, scaler = load_model()
    X = dict_to_array(data)
    X_scaled = scaler.transform(X)
    pred = int(model.predict(X_scaled)[0])
    proba = model.predict_proba(X_scaled)[0]
    confidence = round(float(proba[pred]), 4)
    return {
        "prediction": "Diabetic" if pred == 1 else "Not Diabetic",
        "prediction_code": pred,
        "confidence": confidence,
        "probabilities": {
            "not_diabetic": round(float(proba[0]), 4),
            "diabetic":     round(float(proba[1]), 4),
        },
    }


def predict_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Predict for every row in a DataFrame.
    Adds 'Prediction' and 'Confidence' columns and returns the result.
    """
    model, scaler = load_model()

    # Fill missing columns that are 0-able
    missing_cols = [c for c in FEATURE_NAMES if c not in df.columns]
    if missing_cols:
        raise ValueError(f"CSV is missing columns: {missing_cols}")

    X = df[FEATURE_NAMES].copy()

    # Replace biological zeros with column medians (pandas CoW-safe)
    zero_fill = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    for col in zero_fill:
        if col in X.columns:
            X[col] = X[col].replace(0, np.nan)
            median_val = X[col].median()
            X[col] = X[col].fillna(median_val)

    # Final safety net: fill any remaining NaN with column means
    X = X.fillna(X.mean())

    X_scaled = scaler.transform(X.values)
    preds  = model.predict(X_scaled)
    probas = model.predict_proba(X_scaled)

    result = df.copy()
    result["Prediction"]       = ["Diabetic" if p == 1 else "Not Diabetic" for p in preds]
    result["Prediction_Code"]  = preds.astype(int)
    result["Confidence"]       = [round(float(probas[i][preds[i]]), 4) for i in range(len(preds))]
    result["Prob_Not_Diabetic"]= [round(float(probas[i][0]), 4) for i in range(len(preds))]
    result["Prob_Diabetic"]    = [round(float(probas[i][1]), 4) for i in range(len(preds))]
    return result
