"""Feature engineering for household electricity data."""
import pandas as pd

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract calendar features from an existing datetime column."""
    result = df.copy()
    dt = pd.to_datetime(result["datetime"])
    result["hour"] = dt.dt.hour
    result["day_of_week"] = dt.dt.dayofweek
    result["month"] = dt.dt.month
    result["is_weekend"] = (result["day_of_week"] >= 5).astype(int)
    return result

def add_lag_feature(df: pd.DataFrame, column: str, periods: int) -> pd.DataFrame:
    """Add a historical lag. Use only when data is chronologically ordered."""
    result = df.copy()
    result[f"{column}_lag_{periods}"] = result[column].shift(periods)
    return result
