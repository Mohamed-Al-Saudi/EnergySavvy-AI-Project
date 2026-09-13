"""Rule-based recommendation layer — 5 explainable rules only.

Mirrors notebooks/05_recommendation_system.ipynb:

    Rule 1 - Peak Shift              : hour in [18,19,20,21] and kW > 1.2*daily_mean and kW > 1.5
    Rule 2 - Night Idle Waste        : hour in [1..5] and kW > 0.8
    Rule 3 - Anomaly Inspection      : top 50 rows from df_anom
    Rule 4 - Laundry Load Shift      : laundry share > 40 %
    Rule 5 - Base Load Reduction     : kW > 2.5 for > 100 hours

No LLM. Every recommendation is traceable to a validated threshold,
but every message is written for a human user and refers to real
appliances (AC, water heater, washing machine, dryer, dishwasher, ...).
"""

from __future__ import annotations

import pandas as pd

# ---------- Thresholds (single source of truth) ----------
PEAK_HOURS = [18, 19, 20, 21]
NIGHT_HOURS = [1, 2, 3, 4, 5]

PEAK_KW_MULT = 1.2
PEAK_MIN_KW = 1.5
PEAK_HIGH_KW = 3.0

NIGHT_MIN_KW = 0.8
NIGHT_BASELINE_KW = 0.3

ANOM_TOP_N = 50

SM2_SHARE_THRESHOLD = 40.0
SM2_SAVING_FRACTION = 0.15

BASE_LOAD_KW = 2.5
BASE_LOAD_MIN_HOURS = 100
BASE_LOAD_SAVING_KWH = 0.5

OUTPUT_COLUMNS = [
    "timestamp",
    "type",
    "severity",
    "message",
    "estimated_saving_kwh",
    "condition",
]


# ---------- Helpers ----------
def _prepare(df_h: pd.DataFrame) -> pd.DataFrame:
    """Ensure DatetimeIndex, `hour`, and `daily_mean` exist."""
    df = df_h.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    if "hour" not in df.columns:
        df["hour"] = df.index.hour
    if "daily_mean" not in df.columns:
        df["daily_mean"] = df["Global_active_power"].rolling(24, min_periods=1).mean()
    return df


def _fmt_hour(h: int) -> str:
    """Return 'HH:00' style label."""
    return f"{int(h):02d}:00"


# ---------- Rule 1: Peak Shift ----------
def _rule_peak_shift(df: pd.DataFrame) -> list[dict]:
    recs = []
    peak_df = df[df["hour"].isin(PEAK_HOURS)]
    for idx, row in peak_df.iterrows():
        threshold = row["daily_mean"] * PEAK_KW_MULT
        kw = row["Global_active_power"]
        if kw > threshold and kw > PEAK_MIN_KW:
            recs.append({
                "timestamp": idx,
                "type": "Peak_Shift",
                "severity": "Medium" if kw < PEAK_HIGH_KW else "High",
                "message": (
                    f"Evening peak usage of {kw:.2f} kW detected at "
                    f"{_fmt_hour(row['hour'])}. This is above your typical "
                    f"daily average ({row['daily_mean']:.2f} kW). "
                    f"Shift high-power appliances such as the washing machine, "
                    f"dryer, dishwasher or oven to the off-peak window "
                    f"(13:00–16:00) to cut ~20 % of the tariff cost."
                ),
                "estimated_saving_kwh": round(kw * 0.2, 3),
                "condition": (
                    f"hour={int(row['hour'])} & kW>{threshold:.2f}"
                ),
            })
    return recs


# ---------- Rule 2: Night Idle Waste ----------
def _rule_night_idle(df: pd.DataFrame) -> list[dict]:
    recs = []
    night = df[df["hour"].isin(NIGHT_HOURS)]
    for idx, row in night.iterrows():
        kw = row["Global_active_power"]
        if kw > NIGHT_MIN_KW:
            recs.append({
                "timestamp": idx,
                "type": "Night_Idle_Waste",
                "severity": "Low",
                "message": (
                    f"Unusual nighttime draw of {kw:.2f} kW at "
                    f"{_fmt_hour(row['hour'])} while the household should "
                    f"be idle. Common culprits: Wi-Fi router, refrigerator "
                    f"cycling, TV on standby, phone/laptop chargers, or a "
                    f"water heater keeping temperature overnight. "
                    f"Turn off or unplug standby devices before sleeping."
                ),
                "estimated_saving_kwh": round(
                    max(0.0, kw - NIGHT_BASELINE_KW), 3
                ),
                "condition": "hour in 1-5 & kW>0.8",
            })
    return recs


# ---------- Rule 3: Anomaly Inspection ----------
def _rule_anomaly(df_anom: pd.DataFrame | None) -> list[dict]:
    if df_anom is None or df_anom.empty:
        return []

    recs = []
    for idx, row in df_anom.head(ANOM_TOP_N).iterrows():
        kw = row["Global_active_power"]
        y_pred = row.get("y_pred", float("nan"))
        residual = row.get("residual", float("nan"))
        z = row.get("z_score", 0.0)
        ae = row.get("abs_error", 0.0)

        direction = "higher" if residual > 0 else "lower"

        recs.append({
            "timestamp": idx,
            "type": "Anomaly_Inspection",
            "severity": "High",
            "message": (
                f"Unexpected consumption spike: {kw:.2f} kW observed while "
                f"the model expected around {y_pred:.2f} kW "
                f"({abs(residual):.2f} kW {direction} than expected). "
                f"This often means an appliance was left running — check "
                f"AC, water heater, oven, iron or washing machine for an "
                f"unplanned cycle."
            ),
            "estimated_saving_kwh": round(abs(residual), 3),
            "condition": f"z_score={z:.2f} | abs_error={ae:.2f}",
        })
    return recs


# ---------- Rule 4: Laundry / Dishwasher Shift ----------
def _rule_laundry_shift(df: pd.DataFrame, sub_share_pct: dict | None) -> list[dict]:
    """Sub_metering_2 in this dataset = laundry appliances
    (washing machine, dryer, iron). Sub_metering_1 = kitchen
    (dishwasher, oven, microwave). We report in those terms."""
    if sub_share_pct is None or "Sub_metering_2" not in df.columns:
        return []

    share = sub_share_pct.get("Sub_metering_2", 0)
    if share <= SM2_SHARE_THRESHOLD:
        return []

    return [{
        "timestamp": df.index[-1],
        "type": "Submetering_Shift",
        "severity": "Medium",
        "message": (
            f"Laundry appliances (washing machine, dryer, iron) account "
            f"for {share:.1f} % of your household consumption — above the "
            f"40 % guideline. Try to run full loads instead of partial "
            f"ones and schedule laundry cycles during the off-peak "
            f"window (around 14:00) to lower both energy and tariff cost."
        ),
        "estimated_saving_kwh": round(
            df["Sub_metering_2"].mean() * SM2_SAVING_FRACTION, 3
        ),
        "condition": "laundry share >40%",
    }]


# ---------- Rule 5: High Base Load ----------
def _rule_high_base(df: pd.DataFrame) -> list[dict]:
    high_base = df[df["Global_active_power"] > BASE_LOAD_KW]
    if len(high_base) <= BASE_LOAD_MIN_HOURS:
        return []

    return [{
        "timestamp": df.index[-1],
        "type": "Base_Load_Reduction",
        "severity": "Medium",
        "message": (
            f"Your home has been drawing more than {BASE_LOAD_KW} kW for "
            f"{len(high_base)} hours. This sustained base load usually "
            f"comes from always-on equipment such as the water heater, "
            f"refrigerator, AC running continuously, or several devices "
            f"left on in parallel. Consider energy-efficient replacements "
            f"and avoid running multiple heavy appliances at once."
        ),
        "estimated_saving_kwh": BASE_LOAD_SAVING_KWH,
        "condition": f"kW>{BASE_LOAD_KW} for >{BASE_LOAD_MIN_HOURS} hours",
    }]


# ---------- Public API ----------
def generate_recommendations(
    df_h: pd.DataFrame,
    df_anom: pd.DataFrame | None = None,
    sub_share_pct: dict | None = None,
) -> pd.DataFrame:
    """Run the 5 explainable rules and return a tidy DataFrame.

    Parameters
    ----------
    df_h : hourly DataFrame with DatetimeIndex and 'Global_active_power'
           (optionally 'Sub_metering_2', 'hour', 'daily_mean').
    df_anom : anomaly DataFrame from notebook 04 with columns
              ['Global_active_power', 'y_pred', 'residual',
               'z_score', 'abs_error'] (sorted by severity, top N used).
    sub_share_pct : dict mapping sub-metering column -> % of total,
                    e.g. {'Sub_metering_1': 12.4, 'Sub_metering_2': 47.0,
                          'Sub_metering_3': 40.6}.

    Returns
    -------
    DataFrame with columns:
        timestamp, type, severity, message,
        estimated_saving_kwh, condition
    """
    df = _prepare(df_h)

    recs: list[dict] = []
    recs += _rule_peak_shift(df)
    recs += _rule_night_idle(df)
    recs += _rule_anomaly(df_anom)
    recs += _rule_laundry_shift(df, sub_share_pct)
    recs += _rule_high_base(df)

    if not recs:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    rec_df = (
        pd.DataFrame(recs)
        .drop_duplicates(subset=["timestamp", "type"])
        .sort_values("timestamp")
        .reset_index(drop=True)
    )
    return rec_df


# Backwards-compat alias for old imports
generate_many_recommendations = generate_recommendations
