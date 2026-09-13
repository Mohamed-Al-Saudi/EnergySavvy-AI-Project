"""Egyptian residential tariff helpers used by the live dashboard."""

from __future__ import annotations


# Egyptian residential tiers used in the demo (EGP per kWh).
BILL_TIERS = [
    (350, 1.72),
    (650, 2.18),
    (1000, 2.40),
    (float("inf"), 2.74),
]


def calculate_monthly_bill(monthly_kwh: float) -> float:
    """Return the monthly bill in EGP for a given monthly consumption."""
    kwh = max(0.0, float(monthly_kwh))
    if kwh <= 350:
        rate = 1.72
    elif kwh <= 650:
        rate = 2.18
    elif kwh <= 1000:
        rate = 2.40
    else:
        rate = 2.74
    return round(kwh * rate, 2)


def effective_rate(monthly_kwh: float) -> float:
    """Return the EGP/kWh rate that applies to the household's tier."""
    kwh = max(0.0, float(monthly_kwh))
    if kwh <= 350:
        return 1.72
    if kwh <= 650:
        return 2.18
    if kwh <= 1000:
        return 2.40
    return 2.74


def estimate_monthly_from_devices(simulator, temp_now: float):
    """Estimate daily/monthly kWh and bill from the simulator's device model.

    Mirrors the dashboard-side helper previously inlined in app.py.
    Returns (daily_kwh, monthly_kwh, monthly_bill, rows).
    """
    from datetime import datetime
    import math

    now_hour = datetime.now().hour
    daily_kwh = 0.0
    rows = []

    # Continuous devices: power * 24h * duty-cycle
    for name, cfg in simulator.continuous_devices.items():
        hours = 24.0 * cfg["duty"]
        kwh = cfg["kw"] * hours
        daily_kwh += kwh
        rows.append((name, cfg["kw"], hours, kwh))

    # Variable devices: expected hours from their time/temperature profile
    for name, cfg in simulator.variable_devices.items():
        expected_hours = sum(
            simulator.probability(
                name,
                h,
                temp_now + 2.0 * math.sin((h - now_hour) / 24 * 2 * math.pi),
            )
            for h in range(24)
        )
        kwh = cfg["kw"] * expected_hours
        daily_kwh += kwh
        rows.append((name, cfg["kw"], expected_hours, kwh))

    monthly_kwh = daily_kwh * 30.0
    bill = calculate_monthly_bill(monthly_kwh)
    return daily_kwh, monthly_kwh, bill, rows


def estimate_actual_monthly_bill(current_power_kw: float):
    """Bill if the current instantaneous demand held 24/7 for 30 days."""
    monthly_kwh = max(0.0, float(current_power_kw)) * 24.0 * 30.0
    return monthly_kwh, calculate_monthly_bill(monthly_kwh)


def forecast_cost(forecast_kw_list, baseline_monthly_kwh: float):
    """Return (kwh, cost, effective_rate) for a list of hourly kW forecasts."""
    total_kwh = float(sum(forecast_kw_list))
    rate = effective_rate(baseline_monthly_kwh)
    return total_kwh, round(total_kwh * rate, 2), rate
