"""Anomaly detection helpers."""
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest

def add_anomaly_flag(df: pd.DataFrame, score_column="residual", threshold=3.0) -> pd.DataFrame:
    result = df.copy()
    result["is_anomaly"] = (result[score_column].abs() >= threshold).astype(int)
    return result

def detect_with_residual_and_iso(df, target_col="Global_active_power", pred_col="y_pred"):
    """Real logic from notebook 04"""
    df = df.copy()
    df["residual"] = df[target_col] - df[pred_col]
    window = 24*7
    df["resid_roll_mean"] = df["residual"].rolling(window).mean()
    df["resid_roll_std"] = df["residual"].rolling(window).std()
    df["z_score"] = (df["residual"] - df["resid_roll_mean"]) / df["resid_roll_std"]
    RMSE_proxy = df["residual"].abs().quantile(0.9)
    df["is_anomaly_resid"] = (df["z_score"].abs() > 3.0) | (df["residual"].abs() > 2*RMSE_proxy)

    # IsolationForest part
    feature_cols = [c for c in df.columns if c.startswith(('hour','day','month','lag_','roll_'))]
    if feature_cols:
        iso = IsolationForest(contamination=0.02, random_state=42)
        df["is_anomaly_iso"] = iso.fit_predict(df[feature_cols]) == -1
        df["is_anomaly"] = df["is_anomaly_resid"] | df["is_anomaly_iso"]
    else:
        df["is_anomaly"] = df["is_anomaly_resid"]
    return df
