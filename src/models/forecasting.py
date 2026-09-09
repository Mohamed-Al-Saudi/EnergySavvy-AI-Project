"""Forecasting model helpers."""
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor

def evaluate_forecast(y_true, y_pred) -> dict:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }

def train_forecast_model(X_train, y_train, save_path="models/forecast_rf.pkl"):
    model = RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, save_path)
    return model
