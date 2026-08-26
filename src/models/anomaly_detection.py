"""Anomaly detection helpers.

An anomaly should mean a statistically/model-defined unusual observation,
not automatically a confirmed appliance failure or energy waste.
"""
import pandas as pd

def add_anomaly_flag(df: pd.DataFrame, score_column: str, threshold: float) -> pd.DataFrame:
    """Create a binary anomaly flag from a documented score threshold."""
    result = df.copy()
    result["is_anomaly"] = (result[score_column] >= threshold).astype(int)
    return result
