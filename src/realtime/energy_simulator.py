"""Simulated household electricity telemetry (production IoT stand-in).

Exposes:
    DEVICE_CATALOG              — every household appliance the dashboard can offer
    DEVICE_CATEGORIES           — display categories for grouping in the sidebar
    DEFAULT_CONTINUOUS_DEVICES  — sensible default 24-hour selection
    DEFAULT_VARIABLE_DEVICES    — sensible default time-dependent selection
    HouseholdEnergySimulator    — the simulator used by the dashboard
    EnergySimulator             — legacy alias

Every device in DEVICE_CATALOG carries BOTH:
    * kw       — rated power in kW
    * duty     — duty cycle used when the device is in the 24-hour group
    * profile  — behaviour used when the device is in the time-dependent group
    * category — grouping label for the UI
This lets the sidebar let the user choose either role for any appliance.
"""

from __future__ import annotations

import math
import random
from datetime import datetime


# ===========================================================================
# Full household appliance catalog
# ===========================================================================
DEVICE_CATALOG: dict[str, dict] = {
    # -------------------------------------------------------------------
    # Kitchen / cooking
    # -------------------------------------------------------------------
    "Refrigerator":         {"kw": 0.15, "duty": 0.35, "profile": "always_on", "category": "Kitchen"},
    "Deep freezer":         {"kw": 0.20, "duty": 0.35, "profile": "always_on", "category": "Kitchen"},
    "Freezer":              {"kw": 0.18, "duty": 0.35, "profile": "always_on", "category": "Kitchen"},
    "Stove":                {"kw": 1.20, "duty": 0.05, "profile": "meal",      "category": "Kitchen"},
    "Oven":                 {"kw": 1.50, "duty": 0.03, "profile": "meal",      "category": "Kitchen"},
    "Microwave":            {"kw": 1.20, "duty": 0.02, "profile": "meal",      "category": "Kitchen"},
    "Electric kettle":      {"kw": 1.80, "duty": 0.02, "profile": "meal",      "category": "Kitchen"},
    "Toaster":              {"kw": 1.00, "duty": 0.02, "profile": "meal",      "category": "Kitchen"},
    "Coffee machine":       {"kw": 1.00, "duty": 0.03, "profile": "morning",   "category": "Kitchen"},
    "Blender":              {"kw": 0.50, "duty": 0.01, "profile": "meal",      "category": "Kitchen"},
    "Rice cooker":          {"kw": 0.70, "duty": 0.03, "profile": "meal",      "category": "Kitchen"},
    "Dishwasher":           {"kw": 1.50, "duty": 0.04, "profile": "evening",   "category": "Kitchen"},

    # -------------------------------------------------------------------
    # Laundry
    # -------------------------------------------------------------------
    "Washing machine":      {"kw": 1.20, "duty": 0.03, "profile": "daytime",   "category": "Laundry"},
    "Dryer":                {"kw": 2.50, "duty": 0.03, "profile": "daytime",   "category": "Laundry"},
    "Iron":                 {"kw": 1.20, "duty": 0.02, "profile": "evening",   "category": "Laundry"},

    # -------------------------------------------------------------------
    # Climate control
    # -------------------------------------------------------------------
    "Air conditioner":      {"kw": 1.50, "duty": 0.40, "profile": "ac",        "category": "Climate"},
    "Fans":                 {"kw": 0.075, "duty": 0.30, "profile": "fans",     "category": "Climate"},
    "Space heater":         {"kw": 1.50, "duty": 0.30, "profile": "heater",    "category": "Climate"},
    "Water heater":         {"kw": 1.60, "duty": 0.12, "profile": "water_heater", "category": "Climate"},
    "Dehumidifier":         {"kw": 0.30, "duty": 0.20, "profile": "humidity",  "category": "Climate"},

    # -------------------------------------------------------------------
    # Lighting
    # -------------------------------------------------------------------
    "Lamps":                {"kw": 0.12, "duty": 0.30, "profile": "lamps",     "category": "Lighting"},
    "LED strip":            {"kw": 0.02, "duty": 0.30, "profile": "lamps",     "category": "Lighting"},
    "Garden lights":        {"kw": 0.10, "duty": 0.40, "profile": "night",     "category": "Lighting"},

    # -------------------------------------------------------------------
    # Electronics
    # -------------------------------------------------------------------
    "Wi-Fi router":         {"kw": 0.02, "duty": 1.00, "profile": "always_on", "category": "Electronics"},
    "Television":           {"kw": 0.08, "duty": 0.25, "profile": "evening",   "category": "Electronics"},
    "Laptop":               {"kw": 0.06, "duty": 0.20, "profile": "evening",   "category": "Electronics"},
    "Desktop PC":           {"kw": 0.15, "duty": 0.20, "profile": "evening",   "category": "Electronics"},
    "Gaming console":       {"kw": 0.15, "duty": 0.15, "profile": "evening",   "category": "Electronics"},
    "Phone/device chargers":{"kw": 0.04, "duty": 0.20, "profile": "chargers",  "category": "Electronics"},
    "Radio":                {"kw": 0.03, "duty": 0.15, "profile": "evening",   "category": "Electronics"},
    "Router extender":      {"kw": 0.01, "duty": 1.00, "profile": "always_on", "category": "Electronics"},

    # -------------------------------------------------------------------
    # Personal care
    # -------------------------------------------------------------------
    "Hair dryer":           {"kw": 1.50, "duty": 0.01, "profile": "morning",   "category": "Personal care"},
    "Electric shaver":      {"kw": 0.05, "duty": 0.02, "profile": "morning",   "category": "Personal care"},

    # -------------------------------------------------------------------
    # Cleaning / utility
    # -------------------------------------------------------------------
    "Vacuum cleaner":       {"kw": 0.70, "duty": 0.03, "profile": "vacuum",    "category": "Cleaning"},
    "Water pump":           {"kw": 0.75, "duty": 0.10, "profile": "daytime",   "category": "Cleaning"},
    "Sewing machine":       {"kw": 0.10, "duty": 0.05, "profile": "daytime",   "category": "Cleaning"},

    # -------------------------------------------------------------------
    # Home systems (typically always-on)
    # -------------------------------------------------------------------
    "Security camera":      {"kw": 0.02, "duty": 1.00, "profile": "always_on", "category": "Home systems"},
    "Alarm system":         {"kw": 0.01, "duty": 1.00, "profile": "always_on", "category": "Home systems"},
    "Fish tank pump":       {"kw": 0.03, "duty": 1.00, "profile": "always_on", "category": "Home systems"},
    "Aquarium heater":      {"kw": 0.05, "duty": 0.50, "profile": "always_on", "category": "Home systems"},
}


# Category order for the sidebar
DEVICE_CATEGORIES = [
    "Kitchen",
    "Laundry",
    "Climate",
    "Lighting",
    "Electronics",
    "Personal care",
    "Cleaning",
    "Home systems",
]


# ---------------------------------------------------------------------------
# Default selections shown when the dashboard first loads.
# These reproduce the previous 15-device household so nothing changes by default.
# ---------------------------------------------------------------------------
DEFAULT_CONTINUOUS_DEVICES = [
    "Washing machine",
    "Refrigerator",
    "Stove",
    "Water heater",
    "Wi-Fi router",
    "Radio",
    "Oven",
    "Microwave",
    "Deep freezer",
    "Freezer",
]

DEFAULT_VARIABLE_DEVICES = [
    "Air conditioner",
    "Fans",
    "Lamps",
    "Phone/device chargers",
    "Vacuum cleaner",
]


# ===========================================================================
# Simulator
# ===========================================================================
class HouseholdEnergySimulator:
    """Stateful simulator producing one live reading per `generate()` call.

    Parameters
    ----------
    continuous_devices : dict
        {name: {"kw": float, "duty": float}} — always-available devices.
    variable_devices : dict
        {name: {"kw": float, "profile": str}} — hour/temperature-driven.
    """

    def __init__(self, continuous_devices: dict, variable_devices: dict):
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
        self._previous = {name: False for name in self.variable_devices}

    # ---------------------------------------------------------------
    # Probability model for variable devices
    # ---------------------------------------------------------------
    def probability(self, name: str, hour: int, temp: float) -> float:
        """Hour/temperature-based on-probability for a time-dependent device."""
        profile = self.variable_devices.get(name, {}).get(
            "profile",
            DEVICE_CATALOG.get(name, {}).get("profile", "default"),
        )

        # ---- climate -------------------------------------------------
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

        if profile == "heater":
            if temp <= 12: return 0.75
            if temp <= 16: return 0.55
            if temp <= 20: return 0.30
            return 0.05

        if profile == "humidity":
            # runs a little throughout the day
            return 0.25 if 8 <= hour <= 22 else 0.10

        if profile == "water_heater":
            # morning + evening peaks
            if 6 <= hour <= 9:   return 0.55
            if 18 <= hour <= 22: return 0.65
            return 0.10

        # ---- lighting ------------------------------------------------
        if profile == "lamps":
            if 18 <= hour <= 23 or 0 <= hour <= 5: return 0.78
            if 6 <= hour <= 7: return 0.30
            return 0.035

        if profile == "night":
            if 20 <= hour <= 23 or 0 <= hour <= 2: return 0.70
            return 0.05

        # ---- electronics / household ---------------------------------
        if profile == "chargers":
            if 18 <= hour <= 23: return 0.58
            if 7 <= hour <= 10:  return 0.38
            return 0.10

        if profile == "evening":
            if 18 <= hour <= 23: return 0.65
            if 6 <= hour <= 9:   return 0.25
            return 0.08

        if profile == "morning":
            if 6 <= hour <= 10: return 0.60
            return 0.05

        if profile == "daytime":
            if 9 <= hour <= 17: return 0.45
            return 0.08

        if profile == "meal":
            if 7 <= hour <= 9:   return 0.55
            if 12 <= hour <= 14: return 0.60
            if 18 <= hour <= 21: return 0.70
            return 0.05

        if profile == "vacuum":
            return 0.16 if 9 <= hour <= 17 else 0.01

        # ---- always-on devices, when placed in the variable group ----
        if profile == "always_on":
            return 0.98

        return 0.05

    # ---------------------------------------------------------------
    # One tick
    # ---------------------------------------------------------------
    def generate(self, temperature: float = 30.0, when: datetime | None = None) -> dict:
        """Return one live measurement."""
        now = when or datetime.now()
        hour = now.hour

        voltage = max(215.0, min(245.0,
            230 + 3.0 * math.sin((hour / 24.0) * 2 * math.pi) + random.gauss(0, 1.2)))

        appliances: dict[str, dict] = {}
        total_w = 0.0

        # ---- 24-hour devices: always on, duty-cycle power ----
        for name, cfg in self.continuous_devices.items():
            duty = cfg["duty"]
            active_now = random.random() < min(0.98, 0.12 + duty * 1.8)
            power_kw = cfg["kw"] * random.uniform(0.75, 1.10) if active_now else cfg["kw"] * 0.04
            appliances[name] = {
                "on": True,
                "power_kw": round(power_kw, 3),
                "instant_active": bool(active_now),
                "prob": 1.0,
                "group": "continuous",
            }
            total_w += power_kw * 1000.0

        # ---- Time-dependent devices: switched by hour + temperature ----
        for name, cfg in self.variable_devices.items():
            p = self.probability(name, hour, temperature)
            prev = self._previous.get(name, False)
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
# Legacy alias — default 15-device household
# ---------------------------------------------------------------------------
class EnergySimulator(HouseholdEnergySimulator):
    def __init__(self):
        cont = {n: {"kw": DEVICE_CATALOG[n]["kw"], "duty": DEVICE_CATALOG[n]["duty"]}
                for n in DEFAULT_CONTINUOUS_DEVICES}
        var  = {n: {"kw": DEVICE_CATALOG[n]["kw"], "profile": DEVICE_CATALOG[n]["profile"]}
                for n in DEFAULT_VARIABLE_DEVICES}
        super().__init__(cont, var)
