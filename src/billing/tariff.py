"""Egyptian residential tariff helpers used by the live dashboard."""
from __future__ import annotations

import math
from datetime import datetime


def calculate_monthly_bill(monthly_kwh: float) -> float:
    kwh = max(0.0, float(monthly_kwh))
    if kwh <= 350:   rate = 1.72
    elif kwh <= 650: rate = 2.18
    elif kwh <= 1000: rate = 2.40
    else:            rate = 2.74
    return round(kwh * rate, 2)


def effective_rate(monthly_kwh: float) -> float:
    kwh = max(0.0, float(monthly_kwh))
    if kwh <= 350:   return 1.72
    if kwh <= 650:   return 2.18
    if kwh <= 1000:  return 2.40
    return 2.74


def estimate_monthly_from_devices(simulator, temp_now: float):
    now_hour = datetime.now().hour
    daily_kwh = 0.0
    rows = []
    for name, cfg in simulator.continuous_devices.items():
        hours = 24.0 * cfg["duty"]
        kwh = cfg["kw"] * hours
        daily_kwh += kwh
        rows.append((name, cfg["kw"], hours, kwh))
    for name, cfg in simulator.variable_devices.items():
        expected_hours = sum(
            simulator.probability(
                name, h,
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
    monthly_kwh = max(0.0, float(current_power_kw)) * 24.0 * 30.0
    return monthly_kwh, calculate_monthly_bill(monthly_kwh)


def forecast_cost(forecast_kw_list, baseline_monthly_kwh: float):
    total_kwh = float(sum(forecast_kw_list))
    rate = effective_rate(baseline_monthly_kwh)
    return total_kwh, round(total_kwh * rate, 2), rate
