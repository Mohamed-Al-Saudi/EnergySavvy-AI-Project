"""Utilities for loading the UCI household electricity dataset."""
from pathlib import Path
import pandas as pd

def load_household_data(path: str | Path) -> pd.DataFrame:
    """Load household power data without changing the original source."""
    # Adjust separator and missing-value handling after confirming the raw file format.
    return pd.read_csv(path, sep=";", low_memory=False, na_values="?")
