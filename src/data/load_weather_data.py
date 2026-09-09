"""Utilities for loading Cairo weather data."""
from pathlib import Path
import pandas as pd

def load_weather_data(path: str | Path) -> pd.DataFrame:
    """Load Cairo weather as independent dataset - 5845 rows 2009-2025"""
    df = pd.read_csv(path)
    # auto-detect date column
    date_col = next((c for c in ['time','date','datetime','Date'] if c in df.columns), df.columns[0])
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce', utc=True)
    return df.sort_values(date_col).dropna(subset=[date_col])
