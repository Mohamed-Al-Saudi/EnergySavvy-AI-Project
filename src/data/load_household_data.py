"""Utilities for loading the UCI household electricity dataset."""
from pathlib import Path
import pandas as pd

def load_household_data(path: str | Path) -> pd.DataFrame:
    """Load household power data without changing the original source."""
    # UCI uses ; separator and missing is empty between ;; and sometimes?
    # Confirmed in 01_household_eda: 1.25% missing = 25979 rows
    return pd.read_csv(
        path,
        sep=";",
        low_memory=False,
        na_values=["", "?", "??", "nan"],
        keep_default_na=True
    )
