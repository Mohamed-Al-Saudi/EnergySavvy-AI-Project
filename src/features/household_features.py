"""Feature engineering for household electricity data."""
import pandas as pd

def add_time_features(df: pd.DataFrame, datetime_col="datetime") -> pd.DataFrame:
    result = df.copy()
    dt = pd.to_datetime(result[datetime_col] if datetime_col in result.columns else result.index)
    result["hour"] = dt.hour
    result["dayofweek"] = dt.dayofweek
    result["month"] = dt.month
    result["is_weekend"] = (result["dayofweek"] >= 5).astype(int)
    return result

def add_lag_features(df: pd.DataFrame, column="Global_active_power") -> pd.DataFrame:
    """Same as notebook 03 - chronological lags"""
    result = df.copy()
    for lag in [1,2,3,24,168]: # 1h,2h,3h,24h,1week
        result[f"lag_{lag}"] = result[column].shift(lag)
    result["roll_24_mean"] = result[column].shift(1).rolling(24).mean()
    result["roll_24_std"] = result[column].shift(1).rolling(24).std()
    return result.dropna()

def add_lag_feature(df: pd.DataFrame, column: str, periods: int) -> pd.DataFrame:
    result = df.copy()
    result[f"{column}_lag_{periods}"] = result[column].shift(periods)
    return result
