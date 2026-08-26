"""Reusable preprocessing helpers.

Dataset-specific assumptions should remain explicit. Do not apply the same
cleaning rules blindly to household power and Cairo weather.
"""
import pandas as pd

def remove_exact_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with exact duplicate rows removed."""
    return df.drop_duplicates().copy()

def combine_date_time(df: pd.DataFrame, date_col="date", time_col="time") -> pd.DataFrame:
    """Create a datetime column for household data."""
    result = df.copy()
    result["datetime"] = pd.to_datetime(
        result[date_col].astype(str) + " " + result[time_col].astype(str),
        dayfirst=True,
        errors="coerce",
    )
    return result.sort_values("datetime")
