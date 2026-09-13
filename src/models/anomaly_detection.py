# ----------------------------------------------------------------------
# Live detector — used by dashboard/app.py
# ----------------------------------------------------------------------
def live_anomaly(power_kw, history, z_threshold=3.2, min_abs_deviation=0.45):
    """Flag the current reading as unusual relative to the recent live stream.

    Parameters
    ----------
    power_kw : float
        The current instantaneous reading (kW).
    history : list[float]
        Recent live readings, oldest first. The tail of this list is used
        as the rolling baseline.
    z_threshold : float
        How many standard deviations above the rolling baseline counts as
        an anomaly (default 3.2).
    min_abs_deviation : float
        Minimum absolute kW deviation required to flag an anomaly, so that
        tiny variations during a very quiet window are not flagged.

    Returns
    -------
    (is_anomaly, z, baseline) : tuple[bool, float, float]
    """
    # Not enough history yet — treat as normal.
    if history is None or len(history) < 8:
        baseline = float(np.mean(history)) if history else float(power_kw)
        return False, 0.0, baseline

    arr = np.asarray(history[-40:], dtype=float)

    # Baseline = mean of everything except the newest point.
    past = arr[:-1] if len(arr) > 1 else arr
    baseline = float(np.mean(past))
    std = float(np.std(past)) if len(past) > 2 else 0.08
    std = max(std, 0.06)  # floor so z-score doesn't explode in quiet periods

    z = (float(power_kw) - baseline) / std

    is_anomaly = (
        abs(z) >= z_threshold
        and abs(float(power_kw) - baseline) >= min_abs_deviation
    )
    return bool(is_anomaly), float(z), baseline
