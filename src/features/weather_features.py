"""Feature engineering for Cairo weather analysis.

These features are intentionally independent from household electricity features.
"""
import pandas as pd

def add_temperature_flags(df: pd.DataFrame, temperature_col: str, threshold: float) -> pd.DataFrame:
    """Add a simple high-temperature indicator using a chosen documented threshold."""
    result = df.copy()
    result["is_high_temperature"] = (result[temperature_col] >= threshold).astype(int)
    return result
