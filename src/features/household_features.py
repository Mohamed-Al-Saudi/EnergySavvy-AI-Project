"""Feature engineering for household electricity data."""
from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd


FEATURE_ORDER = [
    "hour", "dayofweek", "month", "is_weekend",
    "lag_1", "lag_2", "lag_3", "lag_24", "lag_168",
    "roll_24_mean", "roll_24_std",
]


# ---------- Batch / notebook layer ----------
def add_time_features(df: pd.DataFrame, datetime_col="datetime") -> pd.DataFrame:
    result = df.copy()
    dt = pd.to_datetime(result[datetime_col] if datetime_col in result.columns else result.index)
    result["hour"] = dt.hour
    result["dayofweek"] = dt.dayofweek
    result["month"] = dt.month
    result["is_weekend"] = (result["dayofweek"] >= 5).astype(int)
    return result


def add_lag_features(df: pd.DataFrame, column="Global_active_power") -> pd.DataFrame:
    result = df.copy()
    for lag in [1, 2, 3, 24, 168]:
        result[f"lag_{lag}"] = result[column].shift(lag)
    result["roll_24_mean"] = result[column].shift(1).rolling(24).mean()
    result["roll_24_std"]  = result[column].shift(1).rolling(24).std()
    return result.dropna()


def add_lag_feature(df: pd.DataFrame, column: str, periods: int) -> pd.DataFrame:
    result = df.copy()
    result[f"{column}_lag_{periods}"] = result[column].shift(periods)
    return result


# ---------- Live / dashboard layer ----------
def build_features_from_values(values, timestamps) -> pd.DataFrame:
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
    f["roll_24_std"]  = f["power_kw"].shift(1).rolling(24).std()
    return f


def create_live_hourly_history(simulator, temp_now: float, hours: int = 336) -> pd.Series:
    """Warm-start an hourly live series from the simulator.

    Default of 336 hours (2 weeks) is required so that lag_168 has enough
    prior history to survive `.dropna()` with a full week of valid rows.
    """
    now = datetime.now().replace(minute=0, second=0, microsecond=0)
    timestamps = [now - timedelta(hours=(hours - 1 - i)) for i in range(hours)]
    vals = []
    for ts in timestamps:
        temp = temp_now + 2.2 * np.sin((ts.hour - now.hour) / 24 * 2 * np.pi)
        vals.append(simulator.generate(temperature=temp, when=ts)["power_kw"])
    return pd.Series(vals, index=pd.DatetimeIndex(timestamps), name="power_kw")

def forecast_live_next_hours(model, hourly_series: pd.Series, hours: int = 6) -> list[float]:
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
            "lag_1": vals[-1], "lag_2": vals[-2], "lag_3": vals[-3],
            "lag_24": vals[-24], "lag_168": vals[-168],
            "roll_24_mean": float(np.mean(vals[-24:])),
            "roll_24_std":  float(np.std(vals[-24:])),
        }
        if model is not None:
            try:
                pred = float(model.predict(np.array([[new_features[k] for k in FEATURE_ORDER]]))[0])
            except Exception:
                pred = float(np.mean(vals[-6:]))
        else:
            pred = float(np.mean(vals[-6:]))
        upper = max(8.0, float(np.percentile(vals, 95)) * 1.5)
        pred = float(np.clip(pred, 0.08, upper))
        preds.append(pred)
        work.loc[next_time] = pred
    return preds
