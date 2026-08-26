"""Utilities for loading Cairo weather data."""
from pathlib import Path
import pandas as pd

def load_weather_data(path: str | Path) -> pd.DataFrame:
    """Load Cairo weather data as an independent dataset."""
    return pd.read_csv(path)
