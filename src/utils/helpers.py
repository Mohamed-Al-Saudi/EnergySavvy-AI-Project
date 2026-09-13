"""General helpers shared across the project."""
from pathlib import Path
import pandas as pd
import json


# ---------------------------------------------------------------------------
# Canonical project paths — single source of truth for every module + the
# deployed dashboard. Add new paths here, not in the callers.
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]   # repo root
DATA_DIR     = PROJECT_ROOT / "data"
MODELS_DIR   = PROJECT_ROOT / "models"
REPORTS_DIR  = PROJECT_ROOT / "reports"
RESULTS_DIR  = REPORTS_DIR / "results"
FIGURES_DIR  = REPORTS_DIR / "figures"
DASH_DIR     = PROJECT_ROOT / "dashboard"
DASH_DATA    = DASH_DIR / "data"

# The trained forecast model used by the live dashboard.
MODEL_PATH   = MODELS_DIR / "forecast_rf.pkl"


# ---------------------------------------------------------------------------
# Filesystem helpers
# ---------------------------------------------------------------------------
def ensure_parent(path):
    """Create parent directories before writing a file - used in 02,03,04,05,06."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    return Path(path)


def ensure_dir(path):
    """Create directory if it does not exist."""
    Path(path).mkdir(parents=True, exist_ok=True)
    return Path(path)


def save_csv(df: pd.DataFrame, path, index=True):
    """Save with parent creation - used for processed/hourly.csv, forecast_metrics.csv, recommendations.csv."""
    ensure_parent(path)
    df.to_csv(path, index=index)
    return path


def save_json(data: dict, path):
    """Save JSON with parent creation - for cairo_weather_summary.json."""
    ensure_parent(path)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    return path


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------
def load_parquet_safe(local_path, github_url=None):
    """Colab-safe loading pattern used in notebooks 03, 04, 05, 06."""
    local_path = Path(local_path)
    if local_path.exists():
        return pd.read_parquet(local_path)
    if github_url:
        print(f"Loading from {github_url}")
        return pd.read_parquet(github_url)
    raise FileNotFoundError(f"{local_path} not found and no github_url provided")


def get_project_root() -> Path:
    """Return the repository root, regardless of where Python is invoked from."""
    return PROJECT_ROOT
