"""Reusable preprocessing helpers."""
import pandas as pd

def remove_exact_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop_duplicates().copy()

def combine_date_time(df: pd.DataFrame, date_col="Date", time_col="Time") -> pd.DataFrame:
    """Create datetime index for household data - format dd/mm/yyyy HH:MM:SS"""
    result = df.copy()
    result["datetime"] = pd.to_datetime(
        result[date_col].astype(str) + " " + result[time_col].astype(str),
        format="%d/%m/%Y %H:%M:%S",
        errors="coerce",
        dayfirst=True
    )
    return result.dropna(subset=["datetime"]).sort_values("datetime").set_index("datetime")

def interpolate_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Time interpolation from notebook 02 - NOT mean. 1.25% sensor outage"""
    return df.interpolate(method='time', limit_direction='both').ffill().bfill()
