"""Forecasting model helpers."""
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

def evaluate_forecast(y_true, y_pred) -> dict:
    """Return standard regression metrics."""
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(root_mean_squared_error(y_true, y_pred)),
    }

# Add model training functions after selecting a baseline and validation strategy.
