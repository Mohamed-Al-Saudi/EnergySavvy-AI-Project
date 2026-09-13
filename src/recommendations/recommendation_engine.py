"""Rule-based recommendation layer — 20 explainable rules.

Mirrors notebooks/05_recommendation_system.ipynb.

Two categories of rules:

    A. Load-shifting opportunities
        R1  Peak_Shift                 hour 18-21, kW > 1.2*daily_mean & > 1.5
        R7  Morning_Peak               hour 7-9,  kW > 1.3*daily_mean & > 1.5
        R8  Off_Peak_Underuse          off-peak avg < 0.6 * overall mean
        R14 Midday_High                hour 11-16, kW > 1.5*daily_mean & > 2.0
        R15 Recurring_Peak_Hour        same hour in top-3 on >40% of days
        R20 Peak_to_Average_Ratio      kW > 3 * overall mean

    B. Waste / baseline / grid-quality issues
        R2  Night_Idle_Waste           hour 1-5, kW > 0.8
        R3  Anomaly_Inspection         top-50 rows from notebook 04
        R4  Submetering_Dominance      any sub-meter share > 40 %
        R5  High_Base_Load             roll_24_mean > 2.5 for > 100 hours
        R6  Weekend_High               weekend_avg > 1.15 * weekday_avg
        R9  Sustained_High_Load        kW > 2.0 for >= 4 consecutive hours
        R10 Low_Voltage_Event          Voltage < 220 V
        R11 High_Current_Warning       Global_intensity > 15 A
        R12 Unmeasured_High            unmeasured_wh > 2000 Wh
        R13 Night_Heavy_Load           hour 0-5, kW > 1.5
        R16 Long_Duration_High         kW > 2.0 for >= 8 consecutive hours
        R17 Consumption_Spike          kW > 2*roll_24_mean + 1
        R18 Very_Low_Consumption       kW < 0.1 for >= 3 consecutive hours
        R19 Monthly_High_Drift         month avg > 1.2 * historical month avg

No LLM. Every recommendation is traceable to a validated threshold.
Messages are written for a human user and mention device *categories*
as possible causes — never asserted as the cause.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd

# ---------- Thresholds (single source of truth) ----------
PEAK_HOURS = [18, 19, 20, 21]
MORNING_PEAK_HOURS = [7, 8, 9]
NIGHT_HOURS = [1, 2, 3, 4, 5]
DEEP_NIGHT_HOURS = [0, 1, 2, 3, 4, 5]
MIDDAY_HOURS = [11, 12, 13, 14, 15, 16]
OFF_PEAK_HOURS = [13, 14, 15, 16]

PEAK_KW_MULT = 1.2
PEAK_MIN_KW = 1.5
PEAK_HIGH_KW = 3.0

MORNING_KW_MULT = 1.3
MIDDAY_KW_MULT = 1.5
MIDDAY_MIN_KW = 2.0

NIGHT_MIN_KW = 0.8
NIGHT_BASELINE_KW = 0.3
NIGHT_HEAVY_KW = 1.5

ANOM_TOP_N = 50

SUB_SHARE_THRESHOLD = 40.0
SUB_SAVING_FRACTION = 0.15

BASE_LOAD_KW = 2.5
BASE_LOAD_MIN_HOURS = 100
BASE_LOAD_SAVING_KWH = 0.5

WEEKEND_HIGH_MULT = 1.15
OFF_PEAK_UNDERUSE_MULT = 0.6

SUSTAINED_KW = 2.0
SUSTAINED_MIN_HOURS = 4
LONG_DURATION_HOURS = 8

SPIKE_MULT = 2.0
SPIKE_OFFSET = 1.0

LOW_VOLTAGE_V = 220.0
HIGH_CURRENT_A = 15.0
UNMEASURED_WH = 2000.0

VERY_LOW_KW = 0.1
VERY_LOW_MIN_HOURS = 3

MONTHLY_DRIFT_MULT = 1.2
PEAK_TO_AVG_MULT = 3.0

RECURRING_TOP_N = 3
RECURRING_MIN_DAY_FRACTION = 0.4

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
    """Ensure DatetimeIndex + derived columns used by rules."""
    df = df_h.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    if "hour" not in df.columns:
        df["hour"] = df.index.hour
    if "daily_mean" not in df.columns:
        df["daily_mean"] = df["Global_active_power"].rolling(24, min_periods=1).mean()
    if "roll_24_mean" not in df.columns:
        df["roll_24_mean"] = (
            df["Global_active_power"].shift(1).rolling(24).mean()
        )
    return df


def _fmt_hour(h: int) -> str:
    return f"{int(h):02d}:00"


def _rec(ts, rtype, severity, message, saving, condition) -> dict:
    return {
        "timestamp": ts,
        "type": rtype,
        "severity": severity,
        "message": message,
        "estimated_saving_kwh": round(float(saving), 3),
        "condition": condition,
    }


def _consecutive_groups(mask: pd.Series) -> list[pd.Index]:
    """Return list of index groups where boolean mask is True consecutively."""
    if mask.empty:
        return []
    m = mask.astype(int)
    gid = (m != m.shift()).cumsum()
    groups = []
    for _, grp in mask.groupby(gid):
        if grp.iloc[0]:
            groups.append(grp.index)
    return groups


# =====================================================================
# A. Load-shifting rules
# =====================================================================
def _rule_peak_shift(df: pd.DataFrame) -> list[dict]:
    recs = []
    for idx, row in df[df["hour"].isin(PEAK_HOURS)].iterrows():
        kw = row["Global_active_power"]
        threshold = row["daily_mean"] * PEAK_KW_MULT
        if kw > threshold and kw > PEAK_MIN_KW:
            recs.append(_rec(
                idx, "Peak_Shift",
                "Medium" if kw < PEAK_HIGH_KW else "High",
                f"Evening peak usage of {kw:.2f} kW at {_fmt_hour(row['hour'])} "
                f"(vs daily average {row['daily_mean']:.2f} kW). Consider moving "
                f"flexible tasks such as laundry, dishwashing or EV charging to "
                f"the off-peak window (13:00-16:00) to reduce tariff cost by "
                f"roughly 20%.",
                kw * 0.2,
                f"hour={int(row['hour'])} & kW>{threshold:.2f}",
            ))
    return recs


def _rule_morning_peak(df: pd.DataFrame) -> list[dict]:
    recs = []
    for idx, row in df[df["hour"].isin(MORNING_PEAK_HOURS)].iterrows():
        kw = row["Global_active_power"]
        threshold = row["daily_mean"] * MORNING_KW_MULT
        if kw > threshold and kw > PEAK_MIN_KW:
            recs.append(_rec(
                idx, "Morning_Peak", "Medium",
                f"Morning spike of {kw:.2f} kW at {_fmt_hour(row['hour'])} "
                f"(daily average {row['daily_mean']:.2f} kW). Several appliances "
                f"starting at the same time — staggering start-up order "
                f"(heating, cooking, water heating) smooths the load and "
                f"avoids peak-rate charges.",
                kw * 0.15,
                f"hour={int(row['hour'])} & kW>{threshold:.2f}",
            ))
    return recs


def _rule_off_peak_underuse(df: pd.DataFrame) -> list[dict]:
    overall = df["Global_active_power"].mean()
    off_peak = df[df["hour"].isin(OFF_PEAK_HOURS)]["Global_active_power"].mean()
    if off_peak < OFF_PEAK_UNDERUSE_MULT * overall:
        return [_rec(
            df.index[-1], "Off_Peak_Underuse", "Low",
            f"Off-peak window (13:00-16:00) only averages {off_peak:.2f} kW vs "
            f"overall {overall:.2f} kW. This cheaper window is under-used. "
            f"Shifting flexible tasks (laundry, dishwashing, EV charging, "
            f"water heating) here cuts the bill without changing comfort.",
            (overall - off_peak) * 4 * 0.2,
            "off_peak_avg<0.6*overall_mean",
        )]
    return []


def _rule_midday_high(df: pd.DataFrame) -> list[dict]:
    recs = []
    for idx, row in df[df["hour"].isin(MIDDAY_HOURS)].iterrows():
        kw = row["Global_active_power"]
        threshold = row["daily_mean"] * MIDDAY_KW_MULT
        if kw > threshold and kw > MIDDAY_MIN_KW:
            recs.append(_rec(
                idx, "Midday_High", "Medium",
                f"Daytime spike of {kw:.2f} kW at {_fmt_hour(row['hour'])} "
                f"(daily average {row['daily_mean']:.2f} kW). A large appliance "
                f"is likely running outside its usual schedule — reviewing "
                f"whether it can be spread over a longer period or scheduled "
                f"for a cheaper window may help.",
                kw * 0.12,
                f"hour={int(row['hour'])} & kW>{threshold:.2f}",
            ))
    return recs


def _rule_recurring_peak_hour(df: pd.DataFrame) -> list[dict]:
    if df.empty:
        return []
    by_hour_by_day = (
        df.groupby([df.index.date, "hour"])["Global_active_power"]
        .mean()
        .unstack()
    )
    if by_hour_by_day.empty:
        return []

    top_per_day = by_hour_by_day.apply(
        lambda r: list(r.sort_values(ascending=False).head(RECURRING_TOP_N).index),
        axis=1,
    ).dropna()
    if top_per_day.empty:
        return []

    flat = [h for lst in top_per_day for h in lst]
    counts = Counter(flat)
    n_days = len(top_per_day)

    recs = []
    for hour_val, count in counts.most_common(RECURRING_TOP_N):
        frac = count / n_days
        if frac > RECURRING_MIN_DAY_FRACTION:
            recs.append(_rec(
                df.index[-1], "Recurring_Peak_Hour", "Medium",
                f"{int(hour_val):02d}:00 is in the top-{RECURRING_TOP_N} "
                f"consumption hours on {count} of {n_days} days "
                f"({frac*100:.0f}%). This is a structural pattern rather than "
                f"a random spike. Adjusting recurring routines at this hour "
                f"gives the largest long-term savings.",
                0.3,
                f"hour={int(hour_val)} appears in top-{RECURRING_TOP_N} "
                f"on {frac*100:.0f}% of days",
            ))
    return recs


def _rule_peak_to_average(df: pd.DataFrame) -> list[dict]:
    overall = df["Global_active_power"].mean()
    if overall <= 0:
        return []
    recs = []
    for idx, row in df.iterrows():
        kw = row["Global_active_power"]
        if kw > PEAK_TO_AVG_MULT * overall:
            recs.append(_rec(
                idx, "Peak_to_Average_Ratio", "High",
                f"Extreme peak of {kw:.2f} kW at {idx:%Y-%m-%d %H:%M} — more "
                f"than {PEAK_TO_AVG_MULT:.0f}x your overall mean of "
                f"{overall:.2f} kW. Very sharp peaks drive demand charges and "
                f"stress the wiring. Consider splitting the load or delaying "
                f"part of it.",
                kw - overall,
                f"kW>{PEAK_TO_AVG_MULT:.0f}*overall_mean",
            ))
    return recs


# =====================================================================
# B. Waste / baseline / grid-quality rules
# =====================================================================
def _rule_night_idle(df: pd.DataFrame) -> list[dict]:
    recs = []
    for idx, row in df[df["hour"].isin(NIGHT_HOURS)].iterrows():
        kw = row["Global_active_power"]
        if kw > NIGHT_MIN_KW:
            recs.append(_rec(
                idx, "Night_Idle_Waste", "Low",
                f"Nighttime idle draw of {kw:.2f} kW at {_fmt_hour(row['hour'])} "
                f"while the house should be quiet. Possible causes: standby / "
                f"always-on devices, a heating or cooling unit cycling, or a "
                f"charger left plugged in. Turning these off can trim the "
                f"overnight baseline.",
                max(0.0, kw - NIGHT_BASELINE_KW),
                "hour in 1-5 & kW>0.8",
            ))
    return recs


def _rule_night_heavy(df: pd.DataFrame) -> list[dict]:
    recs = []
    for idx, row in df[df["hour"].isin(DEEP_NIGHT_HOURS)].iterrows():
        kw = row["Global_active_power"]
        if kw > NIGHT_HEAVY_KW:
            recs.append(_rec(
                idx, "Night_Heavy_Load", "Medium",
                f"Unusually heavy overnight load of {kw:.2f} kW at "
                f"{_fmt_hour(row['hour'])}. Possible causes: a heating or "
                f"cooling unit left on, a water-heating cycle at the wrong "
                f"time, or an appliance running longer than intended. Review "
                f"schedules so only essential items run at night.",
                max(0.0, kw - 0.5) * 0.3,
                f"hour in 0-5 & kW>{NIGHT_HEAVY_KW}",
            ))
    return recs


def _rule_anomaly(df_anom: pd.DataFrame | None) -> list[dict]:
    if df_anom is None or df_anom.empty:
        return []
    recs = []
    for idx, row in df_anom.head(ANOM_TOP_N).iterrows():
        kw = row.get("Global_active_power", float("nan"))
        y_pred = row.get("y_pred", float("nan"))
        residual = row.get("residual", float("nan"))
        z = row.get("z_score", 0.0)
        ae = row.get("abs_error", 0.0)
        if not np.isfinite(residual):
            continue
        direction = "higher" if residual > 0 else "lower"
        recs.append(_rec(
            idx, "Anomaly_Inspection", "High",
            f"Unexpected consumption of {kw:.2f} kW at "
            f"{idx:%Y-%m-%d %H:%M} vs an expected {y_pred:.2f} kW "
            f"(delta {residual:+.2f} kW). Detected as an anomaly in notebook 04. "
            f"Possible causes to check: an appliance left running, an unusual "
            f"load combination, or an unplanned high-power cycle.",
            abs(residual),
            f"z_score={z:.2f} | abs_error={ae:.2f}",
        ))
    return recs


def _rule_submetering_dominance(
    df: pd.DataFrame, sub_share_pct: dict | None
) -> list[dict]:
    if sub_share_pct is None:
        return []
    recs = []
    for col in ["Sub_metering_1", "Sub_metering_2", "Sub_metering_3"]:
        if col not in df.columns:
            continue
        share = sub_share_pct.get(col, 0)
        if share <= SUB_SHARE_THRESHOLD:
            continue

        if col == "Sub_metering_1":
            hint = "a kitchen-related circuit (cooking/cleaning appliances)"
        elif col == "Sub_metering_2":
            hint = "a laundry-related circuit (washing/drying/ironing appliances)"
        else:
            hint = "a heating/cooling-related circuit (water heating or space conditioning)"

        recs.append(_rec(
            df.index[-1], "Submetering_Dominance", "Medium",
            f"One circuit dominates total consumption at {share:.1f}% — "
            f"likely {hint}. Consider staggering its use with other "
            f"high-power tasks and check whether a more efficient appliance "
            f"or off-peak scheduling is feasible.",
            df[col].mean() * SUB_SAVING_FRACTION,
            f"{col} share >{SUB_SHARE_THRESHOLD:.0f}%",
        ))
    return recs


def _rule_high_base(df: pd.DataFrame) -> list[dict]:
    high = df[df["roll_24_mean"] > BASE_LOAD_KW]
    if len(high) <= BASE_LOAD_MIN_HOURS:
        return []
    return [_rec(
        df.index[-1], "High_Base_Load", "Medium",
        f"Average 24h consumption stays above {BASE_LOAD_KW} kW for "
        f"{len(high)} hours. This usually indicates a high always-on "
        f"baseline (heating/cooling, water heating, standby clusters). "
        f"Reducing always-on equipment and grouping flexible loads can "
        f"lower the baseline and the bill.",
        BASE_LOAD_SAVING_KWH,
        f"roll_24_mean>{BASE_LOAD_KW} for >{BASE_LOAD_MIN_HOURS} hours",
    )]


def _rule_weekend_high(df: pd.DataFrame) -> list[dict]:
    if "is_weekend" not in df.columns:
        is_weekend = (df.index.dayofweek >= 5).astype(int)
    else:
        is_weekend = df["is_weekend"]
    wk = df[is_weekend == 0]["Global_active_power"].mean()
    we = df[is_weekend == 1]["Global_active_power"].mean()
    if pd.isna(wk) or pd.isna(we) or wk <= 0:
        return []
    if we > wk * WEEKEND_HIGH_MULT:
        return [_rec(
            df.index[-1], "Weekend_High", "Low",
            f"Weekend consumption averages {we:.2f} kW vs {wk:.2f} kW on "
            f"weekdays ({100*(we/wk-1):.0f}% higher). Weekend-only routines "
            f"drive extra load. Consider spreading heavy tasks (cooking, "
            f"cleaning, laundry) across the day and outside peak hours.",
            (we - wk) * 0.2,
            f"weekend>{wk*WEEKEND_HIGH_MULT:.2f} kW",
        )]
    return []


def _rule_sustained_high(df: pd.DataFrame) -> list[dict]:
    mask = df["Global_active_power"] > SUSTAINED_KW
    recs = []
    for group in _consecutive_groups(mask):
        if len(group) >= SUSTAINED_MIN_HOURS and len(group) < LONG_DURATION_HOURS:
            grp = df.loc[group]
            recs.append(_rec(
                group[0], "Sustained_High_Load", "Medium",
                f"Consumption stayed above {SUSTAINED_KW} kW for "
                f"{len(group)} consecutive hours starting "
                f"{group[0]:%Y-%m-%d %H:%M}. Sustained high load usually "
                f"means a large appliance is running longer than needed, or "
                f"several mid-load devices overlap. Consider shortening "
                f"cycles or staggering them.",
                (grp["Global_active_power"].mean() - SUSTAINED_KW) * len(group) * 0.15,
                f"kW>{SUSTAINED_KW} for {len(group)}h",
            ))
    return recs


def _rule_long_duration_high(df: pd.DataFrame) -> list[dict]:
    mask = df["Global_active_power"] > SUSTAINED_KW
    recs = []
    for group in _consecutive_groups(mask):
        if len(group) >= LONG_DURATION_HOURS:
            grp = df.loc[group]
            recs.append(_rec(
                group[0], "Long_Duration_High", "High",
                f"Consumption remained above {SUSTAINED_KW} kW for "
                f"{len(group)} consecutive hours starting "
                f"{group[0]:%Y-%m-%d %H:%M}. Such extended high load rarely "
                f"reflects an intentional need — inspect for an appliance "
                f"that was never switched off or an automated schedule "
                f"running too often.",
                (grp["Global_active_power"].mean() - SUSTAINED_KW) * len(group) * 0.2,
                f"kW>{SUSTAINED_KW} for {len(group)}h (extended)",
            ))
    return recs


def _rule_low_voltage(df: pd.DataFrame) -> list[dict]:
    if "Voltage" not in df.columns:
        return []
    recs = []
    for idx, row in df[df["Voltage"] < LOW_VOLTAGE_V].iloc[:200].iterrows():
        recs.append(_rec(
            idx, "Low_Voltage_Event", "Medium",
            f"Voltage dropped to {row['Voltage']:.1f} V at "
            f"{idx:%Y-%m-%d %H:%M} (nominal around 230 V). Low voltage often "
            f"coincides with heavy simultaneous loading. Distributing "
            f"high-power appliances across time improves both efficiency "
            f"and appliance lifespan.",
            0.1,
            f"Voltage<{LOW_VOLTAGE_V:.0f}V",
        ))
    return recs


def _rule_high_current(df: pd.DataFrame) -> list[dict]:
    if "Global_intensity" not in df.columns:
        return []
    recs = []
    for idx, row in df[df["Global_intensity"] > HIGH_CURRENT_A].iloc[:200].iterrows():
        recs.append(_rec(
            idx, "High_Current_Warning", "Medium",
            f"Instantaneous current reached {row['Global_intensity']:.1f} A at "
            f"{idx:%Y-%m-%d %H:%M}. This is close to the typical household "
            f"breaker limit. Avoiding simultaneous use of multiple high-power "
            f"appliances reduces the risk of tripping and lowers peak demand "
            f"charges.",
            0.15,
            f"current>{HIGH_CURRENT_A:.0f}A",
        ))
    return recs


def _rule_unmeasured_high(df: pd.DataFrame) -> list[dict]:
    if "unmeasured_wh" not in df.columns:
        return []
    recs = []
    for idx, row in df[df["unmeasured_wh"] > UNMEASURED_WH].iloc[:200].iterrows():
        recs.append(_rec(
            idx, "Unmeasured_High", "Low",
            f"Unmeasured consumption reached {row['unmeasured_wh']:.0f} Wh at "
            f"{idx:%Y-%m-%d %H:%M}. This bucket usually covers lighting, TV, "
            f"computers and small chargers. Efficient lighting and switching "
            f"off idle electronics can cut this category.",
            row["unmeasured_wh"] / 1000 * 0.3,
            f"unmeasured_wh>{UNMEASURED_WH:.0f}",
        ))
    return recs


def _rule_consumption_spike(df: pd.DataFrame) -> list[dict]:
    recs = []
    for idx, row in df.iterrows():
        rmean = row.get("roll_24_mean", np.nan)
        kw = row["Global_active_power"]
        if pd.notna(rmean) and rmean > 0.5 and kw > SPIKE_MULT * rmean + SPIKE_OFFSET:
            recs.append(_rec(
                idx, "Consumption_Spike", "High",
                f"Sharp spike to {kw:.2f} kW at {idx:%Y-%m-%d %H:%M} vs a "
                f"recent 24h average of {rmean:.2f} kW. This is an outlier "
                f"relative to your normal rhythm and often means multiple "
                f"large appliances were started together. Avoiding "
                f"simultaneous start-ups smooths the load.",
                kw - rmean,
                f"kW>{SPIKE_MULT:.0f}*roll_24_mean+{SPIKE_OFFSET:.0f}",
            ))
    return recs


def _rule_very_low_consumption(df: pd.DataFrame) -> list[dict]:
    mask = df["Global_active_power"] < VERY_LOW_KW
    recs = []
    for group in _consecutive_groups(mask):
        if len(group) >= VERY_LOW_MIN_HOURS:
            recs.append(_rec(
                group[0], "Very_Low_Consumption", "Low",
                f"Consumption stayed below {VERY_LOW_KW} kW for "
                f"{len(group)} consecutive hours starting "
                f"{group[0]:%Y-%m-%d %H:%M}. This can mean the house was "
                f"empty — or a meter/sensor gap. Worth verifying so it "
                f"doesn't hide a missed reading or an unexpected outage.",
                0.0,
                f"kW<{VERY_LOW_KW} for {len(group)}h",
            ))
    return recs


def _rule_monthly_high_drift(df: pd.DataFrame) -> list[dict]:
    monthly = df["Global_active_power"].resample("ME").mean()
    if len(monthly) <= 3:
        return []
    last = monthly.iloc[-1]
    hist = monthly.iloc[:-1].mean()
    if pd.isna(hist) or hist <= 0:
        return []
    if last > hist * MONTHLY_DRIFT_MULT:
        return [_rec(
            df.index[-1], "Monthly_High_Drift", "Medium",
            f"The most recent month averages {last:.2f} kW — "
            f"{100*(last/hist-1):.0f}% above the historical average of "
            f"{hist:.2f} kW. Seasonal or behavioural shifts may explain this; "
            f"compare with last year before assuming it's a permanent "
            f"increase.",
            (last - hist) * 24 * 30 * 0.1,
            f"month_avg>{hist*MONTHLY_DRIFT_MULT:.2f} kW",
        )]
    return []


# =====================================================================
# Public API
# =====================================================================
def generate_recommendations(
    df_h: pd.DataFrame,
    df_anom: pd.DataFrame | None = None,
    sub_share_pct: dict | None = None,
) -> pd.DataFrame:
    """Run the 20 explainable rules and return a tidy DataFrame.

    Parameters
    ----------
    df_h : hourly DataFrame with DatetimeIndex and 'Global_active_power'.
           Optionally: Sub_metering_1/2/3, unmeasured_wh, Voltage,
           Global_intensity, hour, daily_mean, roll_24_mean.
    df_anom : anomaly DataFrame from notebook 04 with columns
              ['Global_active_power', 'y_pred', 'residual',
               'z_score', 'abs_error'] (top N rows used).
    sub_share_pct : dict mapping sub-metering column -> % of total,
                    e.g. {'Sub_metering_1': 12.4, 'Sub_metering_2': 47.0,
                          'Sub_metering_3': 40.6}.

    Returns
    -------
    DataFrame with columns:
        timestamp, type, severity, message,
        estimated_saving_kwh, condition
    """
    if df_h is None or df_h.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    if "Global_active_power" not in df_h.columns:
        raise KeyError("df_h must contain a 'Global_active_power' column.")

    df = _prepare(df_h)

    recs: list[dict] = []

    # A. Load-shifting rules
    recs += _rule_peak_shift(df)
    recs += _rule_morning_peak(df)
    recs += _rule_off_peak_underuse(df)
    recs += _rule_midday_high(df)
    recs += _rule_recurring_peak_hour(df)
    recs += _rule_peak_to_average(df)

    # B. Waste / baseline / grid-quality rules
    recs += _rule_night_idle(df)
    recs += _rule_night_heavy(df)
    recs += _rule_anomaly(df_anom)
    recs += _rule_submetering_dominance(df, sub_share_pct)
    recs += _rule_high_base(df)
    recs += _rule_weekend_high(df)
    recs += _rule_sustained_high(df)
    recs += _rule_long_duration_high(df)
    recs += _rule_low_voltage(df)
    recs += _rule_high_current(df)
    recs += _rule_unmeasured_high(df)
    recs += _rule_consumption_spike(df)
    recs += _rule_very_low_consumption(df)
    recs += _rule_monthly_high_drift(df)

    if not recs:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    rec_df = (
        pd.DataFrame(recs)
        .drop_duplicates(subset=["timestamp", "type"])
        .sort_values("timestamp")
        .reset_index(drop=True)
    )
    return rec_df


# Backwards-compat aliases
generate_many_recommendations = generate_recommendations
