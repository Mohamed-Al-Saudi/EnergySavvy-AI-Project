"""Anomaly detection helpers."""
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest


# ----------------------------------------------------------------------
# Simple flag (kept for backwards compatibility)
# ----------------------------------------------------------------------
def add_anomaly_flag(df: pd.DataFrame, score_column="residual", threshold=3.0) -> pd.DataFrame:
    result = df.copy()
    result["is_anomaly"] = (result[score_column].abs() >= threshold).astype(int)
    return result


# ----------------------------------------------------------------------
# Human-readable message builder (matches notebook 04)
# ----------------------------------------------------------------------
def build_anomaly_message(row, target_col="Global_active_power"):
    """Return a general, human-readable explanation for one anomalous hour."""
    kw       = row[target_col]
    y_pred   = row["y_pred"]
    residual = row["residual"]
    z        = row.get("z_score", 0.0)
    hour     = int(row["hour"])
    dow      = int(row["dayofweek"])
    day_str  = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][dow]
    time_str = f"{hour:02d}:00"

    # Which detector(s) fired?
    detectors = []
    if row.get("is_anomaly_resid", False):
        detectors.append("residual z-score")
    if row.get("is_anomaly_iso", False):
        detectors.append("isolation forest")
    det_str = " + ".join(detectors) if detectors else "unknown"

    # Kind of deviation
    if residual > 0:
        kind = "unusual spike"
    elif residual < 0:
        kind = "unusual drop"
    else:
        kind = "unusual pattern"

    # Period of day (context, not a claim)
    if   0 <= hour < 6:   period = "overnight"
    elif 6 <= hour < 10:  period = "morning"
    elif 10 <= hour < 17: period = "daytime"
    elif 17 <= hour < 22: period = "evening peak"
    else:                 period = "late evening"

    # Possible causes — general, framed as things to check
    if period == "overnight":
        causes = ("a device left on overnight, standby/defrost cycles, "
                  "or a heating/cooling unit running at an unusual hour")
    elif period == "morning":
        causes = ("an unusually heavy start-of-day routine, several devices "
                  "switching on at the same time, or an early load not seen in training")
    elif period == "daytime":
        causes = ("a large appliance running outside its usual schedule, "
                  "or an overlap of several medium loads")
    elif period == "evening peak":
        causes = ("more high-power appliances running at once than usual "
                  "during peak hours")
    else:
        causes = ("an unusual combination of loads late at night, "
                  "or a device left running longer than usual")

    return (
        f"[{row.name:%Y-%m-%d %H:%M}] {kind.capitalize()} on {day_str} {time_str} "
        f"({period}) detected by {det_str}: actual {kw:.2f} kW vs expected "
        f"{y_pred:.2f} kW (Δ {residual:+.2f} kW, |z|={abs(z):.2f}). "
        f"Possible causes: {causes}."
    )


# ----------------------------------------------------------------------
# Main detector (matches notebook 04)
# ----------------------------------------------------------------------
def detect_with_residual_and_iso(df, target_col="Global_active_power", pred_col="y_pred"):
    """Real logic from notebook 04 — residual + z-score + Isolation Forest,
    and a human-readable `message` column for every flagged hour."""
    df = df.copy()

    # Residuals
    df["residual"]  = df[target_col] - df[pred_col]
    df["abs_error"] = df["residual"].abs()

    # Rolling z-score (adaptive threshold)
    window = 24 * 7
    df["resid_roll_mean"] = df["residual"].rolling(window).mean()
    df["resid_roll_std"]  = df["residual"].rolling(window).std()
    df["z_score"] = (df["residual"] - df["resid_roll_mean"]) / (df["resid_roll_std"] + 1e-8)

    # Robust threshold — 90th percentile of abs_error
    RMSE_proxy = df["abs_error"].quantile(0.9)
    df["is_anomaly_resid"] = (
        (df["z_score"].abs() > 3.0) | (df["abs_error"] > 2 * RMSE_proxy)
    )

    # IsolationForest part
    feature_cols = [c for c in df.columns
                    if c.startswith(("hour", "day", "month", "lag_", "roll_"))]
    if feature_cols:
        iso = IsolationForest(contamination=0.02, random_state=42)
        df["is_anomaly_iso"] = iso.fit_predict(df[feature_cols]) == -1
        df["is_anomaly"] = df["is_anomaly_resid"] | df["is_anomaly_iso"]
    else:
        df["is_anomaly_iso"] = False
        df["is_anomaly"] = df["is_anomaly_resid"]

    # Message column for every anomalous hour
    df["message"] = ""
    mask = df["is_anomaly"]
    if mask.any():
        df.loc[mask, "message"] = df[mask].apply(
            lambda r: build_anomaly_message(r, target_col=target_col), axis=1
        )

    return df


# ----------------------------------------------------------------------
# Convenience: build the final report (matches notebook 04 report cell)
# ----------------------------------------------------------------------
def build_anomaly_report(df, target_col="Global_active_power"):
    """Return the anomalies sorted by abs_error (most severe first),
    with the columns used by notebook 04 / 05 and the dashboard."""
    report_cols = [target_col, "y_pred", "residual", "z_score", "abs_error",
                   "hour", "dayofweek", "message"]
    # Keep only columns that actually exist (defensive)
    report_cols = [c for c in report_cols if c in df.columns]

    report = (
        df[df["is_anomaly"]][report_cols]
        .sort_values("abs_error", ascending=False)
    )
    return report
