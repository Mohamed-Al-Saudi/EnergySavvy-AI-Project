"""Feature engineering for household electricity data.

Two layers live here:

1. Batch / notebook layer (UCI historical data):
       add_time_features, add_lag_features, add_lag_feature

2. Live / dashboard layer (production simulator):
       build_features_from_values, create_live_hourly_history,
       forecast_live_next_hours, FEATURE_ORDER

Both produce the same feature schema so the trained forecast model can be
used either on historical rows (notebooks) or on live simulated rows (app).
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Feature schema — the exact column order the forecast model expects.
# ---------------------------------------------------------------------------
FEATURE_ORDER = [
    "hour", "dayofweek", "month", "is_weekend",
    "lag_1", "lag_2", "lag_3", "lag_24", "lag_168",
    "roll_24_mean", "roll_24_std",
]


# ===========================================================================
# 1. Batch / notebook layer  (unchanged)
# ===========================================================================
def add_time_features(df: pd.DataFrame, datetime_col: str = "datetime") -> pd.DataFrame:
    """Add hour, dayofweek, month, is_weekend columns."""
    result = df.copy()
    dt = pd.to_datetime(
        result[datetime_col] if datetime_col in result.columns else result.index
    )
    result["hour"] = dt.hour
    result["dayofweek"] = dt.dayofweek
    result["month"] = dt.month
    result["is_weekend"] = (result["dayofweek"] >= 5).astype(int)
    return result


def add_lag_features(df: pd.DataFrame, column: str = "Global_active_power") -> pd.DataFrame:
    """Chronological lags + rolling stats — same as notebook 03."""
    result = df.copy()
    for lag in [1, 2, 3, 24, 168]:  # 1h, 2h, 3h, 24h, 1 week
        result[f"lag_{lag}"] = result[column].shift(lag)
    result["roll_24_mean"] = result[column].shift(1).rolling(24).mean()
    result["roll_24_std"] = result[column].shift(1).rolling(24).std()
    return result.dropna()


def add_lag_feature(df: pd.DataFrame, column: str, periods: int) -> pd.DataFrame:
    """Single lag helper used by earlier experiments."""
    result = df.copy()
    result[f"{column}_lag_{periods}"] = result[column].shift(periods)
    return result


# ===========================================================================
# 2. Live / dashboard layer
# ===========================================================================
def build_features_from_values(values, timestamps) -> pd.DataFrame:
    """Turn a live (values, timestamps) pair into the model's feature frame.

    Parameters
    ----------
    values : sequence of float    — hourly kW readings
    timestamps : sequence of datetime-like
    """
    s = pd.Series(values, index=pd.DatetimeIndex(timestamps), name="power_kw")

    f = pd.DataFrame(index=s.index)
    f["power_kw"] = s
    f["hour"] = f.index.hour
    f["dayofweek"] = f.index.dayofweek
    f["month"] = f.index.month
    f["is_weekend"] = (f["dayofweek"] >= 5).astype(int)

    for lag in [1, 2, 3, 24, 168]:
        f[f"lag_{lag}"] = f["power_kw"].shift(lag)

    f["roll_24_mean"] = f["power_kw"].shift(1).rolling(24).mean()
    f["roll_24_std"] = f["power_kw"].shift(1).rolling(24).std()
    return f


def create_live_hourly_history(simulator, temp_now: float) -> pd.Series:
    """Warm-start a 168-hour hourly series from the LIVE simulator.

    This is a synthetic warm start for the forecast model — it is NOT
    historical UCI data. Every value comes from the simulator that is
    currently running in the dashboard.
    """
    now = datetime.now().replace(minute=0, second=0, microsecond=0)
    timestamps = [now - timedelta(hours=(167 - i)) for i in range(168)]
    vals = []

    for ts in timestamps:
        # Approximate the current live temperature across the simulated day.
        temp = temp_now + 2.2 * np.sin((ts.hour - now.hour) / 24 * 2 * np.pi)
        d = simulator.generate(temperature=temp, when=ts)
        vals.append(d["power_kw"])

    return pd.Series(vals, index=pd.DatetimeIndex(timestamps), name="power_kw")


def forecast_live_next_hours(model, hourly_series: pd.Series, hours: int = 6) -> list[float]:
    """Recursively forecast the next `hours` values using the live series.

    If the model is None (offline / model file missing), falls back to a
    short moving average of the last few readings.
    """
    work = hourly_series.copy()
    preds: list[float] = []

    for _ in range(hours):
        next_time = work.index[-1] + timedelta(hours=1)
        vals = list(work.values)

        new_features = {
            "hour": next_time.hour,
            "dayofweek": next_time.dayofweek,
            "month": next_time.month,
            "is_weekend": int(next_time.dayofweek >= 5),
            "lag_1": vals[-1],
            "lag_2": vals[-2],
            "lag_3": vals[-3],
            "lag_24": vals[-24],
            "lag_168": vals[-168],
            "roll_24_mean": float(np.mean(vals[-24:])),
            "roll_24_std": float(np.std(vals[-24:])),
        }

        if model is not None:
            try:
                pred = float(
                    model.predict(
                        np.array([[new_features[k] for k in FEATURE_ORDER]])
                    )[0]
                )
            except Exception:
                pred = float(np.mean(vals[-6:]))
        else:
            pred = float(np.mean(vals[-6:]))

        # Keep the forecast within a plausible range around the live series.
        upper = max(8.0, float(np.percentile(vals, 95)) * 1.5)
        pred = float(np.clip(pred, 0.08, upper))
        preds.append(pred)

        work.loc[next_time] = pred

    return preds
