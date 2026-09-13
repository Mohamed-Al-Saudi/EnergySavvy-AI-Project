"""Simulated household electricity telemetry.

Production stand-in for future IoT / smart-meter sensors.

Exposes:
    DEFAULT_CONTINUOUS_DEVICES  — list of 24-hour device names
    DEFAULT_VARIABLE_DEVICES    — list of time-dependent device names
    DEVICE_CATALOG              — dict: name -> {kw, duty?, profile?}
    HouseholdEnergySimulator    — the simulator used by the dashboard

Also keeps a legacy `EnergySimulator` alias for older code/tests.
"""

from __future__ import annotations

import math
import random
from datetime import datetime


# ---------------------------------------------------------------------------
# Device catalog
# ---------------------------------------------------------------------------
# Every device appears exactly once. Continuous devices carry `duty`
# (fraction of the day they actually draw power). Variable devices carry a
# `profile` that decides their hour/temperature-dependent probability.

DEVICE_CATALOG = {
    # ---- 24-hour household devices (always available, duty-cycled) ----
    "Washing machine":        {"kw": 1.20, "duty": 0.03, "group": "continuous"},
    "Refrigerator":           {"kw": 0.15, "duty": 0.35, "group": "continuous"},
    "Stove":                  {"kw": 1.20, "duty": 0.05, "group": "continuous"},
    "Water heater":           {"kw": 1.60, "duty": 0.12, "group": "continuous"},
    "Wi-Fi router":           {"kw": 0.02, "duty": 1.00, "group": "continuous"},
    "Radio":                  {"kw": 0.03, "duty": 0.15, "group": "continuous"},
    "Oven":                   {"kw": 1.50, "duty": 0.03, "group": "continuous"},
    "Microwave":              {"kw": 1.20, "duty": 0.02, "group": "continuous"},
    "Deep freezer":           {"kw": 0.20, "duty": 0.35, "group": "continuous"},
    "Freezer":                {"kw": 0.18, "duty": 0.35, "group": "continuous"},

    # ---- Time-dependent devices (switched by hour + temperature) ----
    "Air conditioner":        {"kw": 1.50, "profile": "ac",       "group": "variable"},
    "Fans":                   {"kw": 0.075, "profile": "fans",    "group": "variable"},
    "Lamps":                  {"kw": 0.12, "profile": "lamps",    "group": "variable"},
    "Phone/device chargers":  {"kw": 0.04, "profile": "chargers", "group": "variable"},
    "Vacuum cleaner":         {"kw": 0.70, "profile": "vacuum",   "group": "variable"},
}

DEFAULT_CONTINUOUS_DEVICES = [
    name for name, cfg in DEVICE_CATALOG.items() if cfg["group"] == "continuous"
]

DEFAULT_VARIABLE_DEVICES = [
    name for name, cfg in DEVICE_CATALOG.items() if cfg["group"] == "variable"
]


# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------
class HouseholdEnergySimulator:
    """Stateful simulator producing one live reading per `generate()` call.

    Parameters
    ----------
    continuous_devices : dict
        {name: {"kw": float, "duty": float}} — always-available devices.
    variable_devices : dict
        {name: {"kw": float, "profile": str}} — time/temperature-dependent.
    """

    def __init__(self, continuous_devices: dict, variable_devices: dict):
        # Store configs; freeze so the bill module can iterate safely.
        self.continuous_devices = {
            name: {"kw": float(cfg["kw"]), "duty": float(cfg["duty"])}
            for name, cfg in continuous_devices.items()
        }
        self.variable_devices = {
            name: {
                "kw": float(cfg["kw"]),
                "profile": cfg.get(
                    "profile",
                    DEVICE_CATALOG.get(name, {}).get("profile", "default"),
                ),
            }
            for name, cfg in variable_devices.items()
        }
        # Track previous state so variable devices have realistic "sticky" behaviour.
        self._previous = {name: False for name in self.variable_devices}

    # ---------------------------------------------------------------
    # Probability model for variable devices
    # ---------------------------------------------------------------
    def probability(self, name: str, hour: int, temp: float) -> float:
        """Hour/temperature-based on-probability for a variable device."""
        profile = self.variable_devices.get(name, {}).get(
            "profile",
            DEVICE_CATALOG.get(name, {}).get("profile", "default"),
        )

        if profile == "ac":
            if temp >= 34:   base = 0.90
            elif temp >= 31: base = 0.72
            elif temp >= 28: base = 0.48
            elif temp >= 25: base = 0.24
            else:            base = 0.06
            if 0 <= hour <= 6:
                base *= 0.55
            return min(base, 0.95)

        if profile == "fans":
            if temp >= 32: return 0.72
            if temp >= 29: return 0.52
            if temp >= 25: return 0.30
            return 0.07

        if profile == "lamps":
            if 18 <= hour <= 23 or 0 <= hour <= 5: return 0.78
            if 6 <= hour <= 7:                     return 0.30
            return 0.035

        if profile == "chargers":
            if 18 <= hour <= 23: return 0.58
            if 7 <= hour <= 10:  return 0.38
            return 0.10

        if profile == "vacuum":
            return 0.16 if 9 <= hour <= 17 else 0.01

        return 0.05

    # ---------------------------------------------------------------
    # One tick
    # ---------------------------------------------------------------
    def generate(self, temperature: float = 30.0, when: datetime | None = None) -> dict:
        """Return one live measurement.

        Output keys
        -----------
        timestamp : ISO string
        voltage_v, current_a, power_kw, power_w : float
        appliances : {name: {on, power_kw, instant_active, prob, group}}
        """
        now = when or datetime.now()
        hour = now.hour

        # Smooth daily voltage swing + small noise.
        voltage = max(
            215.0,
            min(
                245.0,
                230 + 3.0 * math.sin((hour / 24.0) * 2 * math.pi)
                + random.gauss(0.0, 1.2),
            ),
        )

        appliances: dict[str, dict] = {}
        total_w = 0.0

        # ---- Continuous devices: always on, duty-cycle power ----
        for name, cfg in self.continuous_devices.items():
            duty = cfg["duty"]
            # Compressor/motor style: instant activity is short pulses around duty.
            active_now = random.random() < min(0.98, 0.12 + duty * 1.8)
            if active_now:
                power_kw = cfg["kw"] * random.uniform(0.75, 1.10)
            else:
                power_kw = cfg["kw"] * 0.04

            appliances[name] = {
                "on": True,
                "power_kw": round(power_kw, 3),
                "instant_active": bool(active_now),
                "prob": 1.0,
                "group": "continuous",
            }
            total_w += power_kw * 1000.0

        # ---- Variable devices: switched by hour + temperature ----
        for name, cfg in self.variable_devices.items():
            p = self.probability(name, hour, temperature)
            prev = self._previous.get(name, False)
            # Sticky behaviour: an already-running device is likely to keep running.
            stay_p = min(0.96, p + 0.35) if prev else p
            is_on = random.random() < stay_p
            power_kw = cfg["kw"] * random.uniform(0.80, 1.15) if is_on else 0.0

            appliances[name] = {
                "on": bool(is_on),
                "power_kw": round(power_kw, 3),
                "instant_active": bool(is_on),
                "prob": round(float(p), 2),
                "group": "variable",
            }
            self._previous[name] = is_on
            total_w += power_kw * 1000.0

        # Floor so the dashboard never shows a flat zero household.
        total_w = max(120.0, total_w)

        return {
            "timestamp": now.isoformat(),
            "voltage_v": round(float(voltage), 1),
            "current_a": round(float(total_w / voltage), 2),
            "power_kw": round(total_w / 1000.0, 3),
            "power_w": round(float(total_w), 1),
            "appliances": appliances,
            "is_simulated": True,
            "source": "Temporary IoT simulator",
        }


# ---------------------------------------------------------------------------
# Legacy alias — old EnergySimulator with the same fixed device set.
# Keeps older imports working without changing call sites.
# ---------------------------------------------------------------------------
def _legacy_defaults() -> tuple[dict, dict]:
    cont = {
        name: {"kw": DEVICE_CATALOG[name]["kw"], "duty": DEVICE_CATALOG[name]["duty"]}
        for name in DEFAULT_CONTINUOUS_DEVICES
    }
    var = {
        name: {"kw": DEVICE_CATALOG[name]["kw"], "profile": DEVICE_CATALOG[name]["profile"]}
        for name in DEFAULT_VARIABLE_DEVICES
    }
    return cont, var


class EnergySimulator(HouseholdEnergySimulator):
    """Backwards-compatible alias using the default 15-device household."""

    def __init__(self):
        cont, var = _legacy_defaults()
        super().__init__(cont, var)
