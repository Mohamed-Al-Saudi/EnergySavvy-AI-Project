"""Forecasting model helpers.

Two layers:

1. Training layer (notebooks 03):
       train_forecast_model, evaluate_forecast

2. Deployment layer (dashboard):
       load_model  → load the trained RandomForest used by the live app
       predict     → run the model on a feature frame
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.utils.helpers import MODEL_PATH


# ===========================================================================
# Training / evaluation (notebooks)
# ===========================================================================
def evaluate_forecast(y_true, y_pred) -> dict:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }


def train_forecast_model(X_train, y_train, save_path="models/forecast_rf.pkl"):
    model = RandomForestRegressor(
        n_estimators=100, max_depth=12, random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_train)
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, save_path)
    return model


# ===========================================================================
# Deployment helpers (dashboard)
# ===========================================================================
def load_model(path=None):
    """Load the trained forecast model.

    Parameters
    ----------
    path : str | Path | None
        Override path. Defaults to `src.utils.helpers.MODEL_PATH`.

    Returns
    -------
    sklearn model or None
        Returns None if the model file is missing or fails to load; the
        dashboard then falls back to a moving-average forecast.
    """
    model_path = Path(path) if path is not None else MODEL_PATH
    if not model_path.exists():
        return None
    try:
        return joblib.load(model_path)
    except Exception:
        return None


def predict(model, features):
    """Predict with a loaded model; returns a numpy array of predictions."""
    return model.predict(features)
