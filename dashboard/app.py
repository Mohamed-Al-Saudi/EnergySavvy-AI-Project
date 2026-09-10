"""
EnergySavvy AI
---------------
Final Streamlit deployment application.

Pipeline:
    Live Location
        -> Open-Meteo Weather
        -> RealtimeEngine
        -> EnergySimulator (stateful, realism-corrected — see note below)
        -> Forecast Model (forecast_rf.pkl)
        -> Anomaly Detection
        -> Recommendations
        -> Streamlit Dashboard

IMPORTANT — training data vs. live data:
    The UCI Household Power Consumption dataset and the Cairo Weather
    dataset were used ONLY to train forecast_rf.pkl (see notebooks/).
    Nothing in this file reads those historical CSVs. Everything shown
    here is generated live: real weather, a real, presenter-selected
    Egyptian location, and a live household simulator whose output is
    fed straight into the trained model.

NOTE on the appliance simulator:
    The original EnergySimulator.generate() draws each appliance's
    on/off state independently on every call, with no memory of the
    previous state. In practice this meant that, at any given instant,
    only the appliances with a high standalone probability (mainly the
    refrigerator) were ever shown "on" — real appliances do not behave
    this way. A refrigerator and router run continuously; an air
    conditioner, once triggered by heat, keeps running for a while; a
    kettle or vacuum cleaner runs briefly and then stops. To reflect
    that, this file implements a stateful appliance model (persistence
    / hysteresis across ticks, categorized by usage pattern) that is
    used for both the live feed and the historical backfill below.
    EnergySimulator itself is left untouched in src/.

Run locally (from the project root):
    streamlit run dashboard/app.py
"""

import io
import math
import random
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ============================================================
# PROJECT PATHS
# ============================================================

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

FORECAST_MODEL_PATH = PROJECT_ROOT / "models" / "forecast_rf.pkl"

# ============================================================
# IMPORT EXISTING PROJECT MODULES
# ============================================================

from src.realtime.realtime_engine import RealtimeEngine
from src.realtime.inference_engine import InferenceEngine
from src.realtime.weather_api import get_current_weather, get_location_by_ip

try:
    from streamlit_autorefresh import st_autorefresh
    HAS_AUTOREFRESH = True
except ImportError:
    HAS_AUTOREFRESH = False

try:
    import qrcode
    HAS_QRCODE = True
except ImportError:
    HAS_QRCODE = False


REFRESH_INTERVAL_SECONDS = 8
HISTORY_BACKFILL_HOURS = 168  # One week of hourly points, matches lag_168
CHART_HISTORY_POINTS = 60

# Major Egyptian cities with accurate coordinates, so the live weather
# reflects the presenter's actual demo location rather than wherever
# the hosting server happens to be. IP-based geolocation is offered as
# a secondary option, with a clear caveat (see sidebar).
# All 27 Egyptian governorates. Coordinates point to each governorate's
# main city/capital and are used only to request live weather from Open-Meteo.
# The dashboard therefore supports a complete Egypt-wide demo, not only
# the original Cairo/major-city shortlist.
EGYPT_GOVERNORATES = {
    "Cairo": (30.0444, 31.2357),
    "Alexandria": (31.2001, 29.9187),
    "Port Said": (31.2653, 32.3019),
    "Suez": (29.9668, 32.5498),
    "Damietta": (31.4175, 31.8144),
    "Dakahlia": (31.0409, 31.3785),
    "Sharqia": (30.5877, 31.5020),
    "Qalyubia": (30.4598, 31.1786),
    "Kafr El Sheikh": (31.1107, 30.9388),
    "Gharbia": (30.7865, 31.0004),
    "Monufia": (30.5972, 30.9876),
    "Beheira": (31.0341, 30.4682),
    "Ismailia": (30.5965, 32.2715),
    "Giza": (30.0131, 31.2089),
    "Fayoum": (29.3084, 30.8428),
    "Beni Suef": (29.0661, 31.0994),
    "Minya": (28.1099, 30.7503),
    "Assiut": (27.1809, 31.1837),
    "Sohag": (26.5591, 31.6957),
    "Qena": (26.1551, 32.7160),
    "Luxor": (25.6872, 32.6396),
    "Aswan": (24.0889, 32.8998),
    "Red Sea": (27.2579, 33.8116),
    "New Valley": (25.4410, 30.5586),
    "Matrouh": (31.3543, 27.2373),
    "North Sinai": (31.1313, 33.8033),
    "South Sinai": (28.2173, 33.6254),
}

# Backward-compatible alias for any code that may still reference the old name.
EGYPT_CITIES = EGYPT_GOVERNORATES

# EgyptERA residential tariff currently published for 2026.
# IMPORTANT: residential consumption above 1,000 kWh is billed at the
# published residential rate; 2.74 EGP/kWh is associated with the
# separate code-meter tariff, not the normal residential tier schedule.
TARIFF_TIERS_EGP_PER_KWH = [
    (50, 0.68),
    (100, 0.78),
    (200, 0.95),
    (350, 1.55),
    (650, 1.95),
    (1000, 2.10),
    (float("inf"), 2.58),
]

# Fixed monthly customer-service charge by residential consumption band.
SERVICE_FEES_EGP = [
    (50, 1.0),
    (100, 2.0),
    (200, 6.0),
    (350, 11.0),
    (650, 15.0),
    (1000, 25.0),
    (float("inf"), 40.0),
]

# The instantaneous simulator power is intentionally NOT multiplied by 24h
# and projected directly into a bill. Several appliances are represented as
# continuously available/connected, but their real energy use is duty-cycled
# (compressor cycles, thermostat cycles, standby/background consumption,
# occasional operation). This keeps the monthly projection physically
# plausible while preserving the requested 24H device category in the UI.
CONTINUOUS_DAILY_DUTY = {
    "refrigerator": 0.35,
    "washing_machine": 0.03,
    "stove": 0.05,
    "water_heater": 0.12,
    "router_modem": 1.00,
    "radio": 0.15,
    "oven": 0.03,
    "microwave": 0.02,
    "deep_freezer": 0.35,
    "freezer": 0.35,
}

VARIABLE_USAGE_DAYS = 30


def calculate_bill_egp(monthly_kwh: float) -> float:
    """Calculate a residential bill from monthly consumption using the
    currently published EgyptERA 2026 residential tiers plus the
    corresponding customer-service charge."""

    if monthly_kwh <= 0:
        return 0.0

    remaining = float(monthly_kwh)
    cost = 0.0
    previous_cap = 0.0

    for cap, rate in TARIFF_TIERS_EGP_PER_KWH:
        band_width = remaining if cap == float("inf") else min(remaining, cap - previous_cap)
        if band_width > 0:
            cost += band_width * rate
            remaining -= band_width
        previous_cap = cap
        if remaining <= 0:
            break

    service_fee = SERVICE_FEES_EGP[-1][1]
    for cap, fee in SERVICE_FEES_EGP:
        if monthly_kwh <= cap:
            service_fee = fee
            break

    return cost + service_fee


def estimate_monthly_usage(appliances: dict, anchor_temperature: float, anchor_hour: int) -> dict:
    """Estimate future monthly energy from appliance operating behavior.

    This is deliberately schedule-based instead of using:
        current_instantaneous_kw * 24 * 30

    The latter can turn a short-lived simultaneous load into an unrealistic
    month-long load. Continuous devices retain their requested 24H status in
    the simulator, while their billing contribution uses realistic duty
    factors. Variable devices use their time/temperature probabilities across
    a synthetic day.
    """

    daily_kwh = 0.0
    rows = []

    for name, config in APPLIANCES.items():
        rated_kw = float(config["rated_kw"])

        if config["category"] == "continuous":
            duty = CONTINUOUS_DAILY_DUTY.get(name, 0.10)
            expected_hours = 24.0 * duty
        else:
            expected_hours = sum(
                _target_probability(
                    name,
                    hour,
                    _synthetic_hour_temperature(hour, anchor_temperature, anchor_hour),
                )
                for hour in range(24)
            )

        daily_device_kwh = rated_kw * expected_hours
        daily_kwh += daily_device_kwh
        rows.append({
            "Appliance": name.replace("_", " ").title(),
            "Group": config["group"],
            "Expected hours/day": round(expected_hours, 2),
            "Estimated daily kWh": round(daily_device_kwh, 2),
        })

    monthly_kwh = daily_kwh * VARIABLE_USAGE_DAYS
    bill = calculate_bill_egp(monthly_kwh)

    return {
        "daily_kwh": daily_kwh,
        "monthly_kwh": monthly_kwh,
        "monthly_bill_egp": bill,
        "effective_rate": bill / monthly_kwh if monthly_kwh else 0.0,
        "rows": rows,
    }


# ============================================================
# STREAMLIT PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="EnergySavvy AI",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# STYLE — dark, professional, with a slow-moving gradient backdrop
# and glass-panel cards so the interface reads as active rather than
# static, without sacrificing formality.
# ============================================================

st.markdown(
    """
    <style>

    html, body, [class*="css"] {
        font-family: "Segoe UI", "Inter", "Helvetica Neue", Arial, sans-serif;
    }

    :root {
        --accent-1: #8b5cf6;
        --accent-2: #ec4899;
        --accent-3: #f97316;
        --accent-4: #22b8cf;
    }

    /* Stable competition background: intentionally NO background animation.
       Streamlit reruns the page every few seconds; an animated background
       restarts on every rerun and can briefly flash almost black. */
    .stApp {
        background-color: #101329;
        background-image:
            linear-gradient(115deg, #30205a 0%, #17182d 38%, #0d1c2d 68%, #073d4a 100%),
            radial-gradient(circle at 8% 8%, rgba(139, 92, 246, 0.34), transparent 34%),
            radial-gradient(circle at 92% 10%, rgba(34, 184, 207, 0.28), transparent 36%),
            radial-gradient(circle at 50% 95%, rgba(236, 72, 153, 0.14), transparent 42%);
        background-attachment: fixed;
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
    }

    [data-testid="stAppViewContainer"],
    [data-testid="stMain"] {
        background: transparent;
    }

    [data-testid="stHeader"] {
        background: rgba(9, 12, 25, 0.18);
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(25, 19, 50, 0.98), rgba(7, 27, 39, 0.98));
        border-right: 1px solid rgba(139, 92, 246, 0.24);
    }

    .eyebrow {
        font-size: 12px;
        letter-spacing: 3px;
        text-transform: uppercase;
        color: #9aa4b8;
        font-weight: 600;
        margin-bottom: 4px;
    }

    .main-title {
        font-size: 42px;
        font-weight: 800;
        letter-spacing: -0.5px;
        margin-bottom: 4px;
        background: linear-gradient(100deg, var(--accent-1), var(--accent-2) 55%, var(--accent-3));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .subtitle {
        font-size: 15px;
        color: #a3aec2;
        margin-bottom: 20px;
        display: flex;
        align-items: center;
    }

    .project-info {
        display: inline-block;
        padding: 8px 13px;
        margin: -8px 0 14px 0;
        border: 1px solid rgba(139, 92, 246, 0.24);
        border-radius: 12px;
        background: rgba(10, 14, 28, 0.38);
        color: #cbd5e1;
        font-size: 12px;
        line-height: 1.6;
        backdrop-filter: blur(8px);
    }


    .footer-card {
        margin-top: 8px;
        padding: 18px 20px;
        border: 1px solid rgba(139, 92, 246, 0.24);
        border-radius: 16px;
        background: linear-gradient(135deg, rgba(42, 28, 72, 0.62), rgba(8, 31, 43, 0.62));
        backdrop-filter: blur(10px);
    }

    .footer-title { font-size: 18px; font-weight: 700; color: #f4f7fb; margin-bottom: 8px; }
    .footer-line { color: #aeb7c8; font-size: 12px; line-height: 1.8; }
    .footer-line a { color: #67e8f9; text-decoration: none; }
    .footer-line a:hover { text-decoration: underline; }
    .footer-description { margin-top: 9px; color: #8f9bb0; font-size: 11px; line-height: 1.7; max-width: 1100px; }

    .live-dot {
        height: 9px;
        width: 9px;
        background-color: #22c55e;
        border-radius: 50%;
        display: inline-block;
        margin-right: 10px;
        animation: pulse 1.8s infinite;
        flex-shrink: 0;
    }

    @keyframes pulse {
        0%   { box-shadow: 0 0 0 0 rgba(34, 197, 94, 0.55); }
        70%  { box-shadow: 0 0 0 9px rgba(34, 197, 94, 0); }
        100% { box-shadow: 0 0 0 0 rgba(34, 197, 94, 0); }
    }

    .pipeline-strip {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin: 8px 0 18px;
    }

    .pipeline-chip {
        display: inline-block;
        padding: 6px 12px;
        border-radius: 999px;
        font-size: 10px;
        letter-spacing: 1px;
        font-weight: 800;
        border: 1px solid rgba(255,255,255,0.14);
        background: rgba(255,255,255,0.035);
    }

    .pipeline-chip.live { color: #4ade80; border-color: rgba(34,197,94,0.45); }
    .pipeline-chip.weather { color: #22d3ee; border-color: rgba(34,211,238,0.40); }
    .pipeline-chip.power { color: #fb923c; border-color: rgba(251,146,60,0.42); }
    .pipeline-chip.ai { color: #c084fc; border-color: rgba(192,132,252,0.42); }

    .section-title {
        font-size: 21px;
        font-weight: 700;
        color: #eef1f7;
        margin-top: 34px;
        margin-bottom: 4px;
        border-left: 3px solid var(--accent-4);
        padding-left: 12px;
    }

    .section-caption {
        font-size: 13px;
        color: #8b96ab;
        margin-bottom: 16px;
        padding-left: 15px;
    }

    /* Glass panel cards */
    .glass-card {
        background: linear-gradient(155deg, rgba(255,255,255,0.055), rgba(255,255,255,0.015));
        border: 1px solid rgba(255,255,255,0.09);
        border-radius: 14px;
        padding: 20px 22px;
        height: 100%;
        transition: border-color 0.3s ease, transform 0.3s ease;
    }

    .glass-card:hover {
        border-color: rgba(139, 92, 246, 0.5);
        transform: translateY(-2px);
    }

    .card-label {
        font-size: 12px;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        color: #8b96ab;
        font-weight: 700;
        margin-bottom: 10px;
    }

    .card-value {
        font-size: 30px;
        font-weight: 800;
        line-height: 1.15;
        background: linear-gradient(90deg, var(--accent-1), var(--accent-2));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .card-value.plain {
        background: none;
        -webkit-text-fill-color: #eef1f7;
        color: #eef1f7;
    }

    .card-sub {
        font-size: 12.5px;
        color: #9aa4b8;
        margin-top: 8px;
    }

    .kpi-row {
        display: flex;
        gap: 16px;
        flex-wrap: wrap;
        margin-bottom: 6px;
    }

    .kpi-row > div {
        flex: 1;
        min-width: 190px;
    }

    /* Status pills */
    .pill {
        display: inline-block;
        padding: 3px 12px;
        border-radius: 999px;
        font-size: 12.5px;
        font-weight: 700;
        letter-spacing: 0.3px;
    }

    .pill-positive {
        background-color: rgba(34, 197, 94, 0.16);
        color: #4ade80;
        border: 1px solid rgba(34, 197, 94, 0.35);
    }

    .pill-negative {
        background-color: rgba(239, 68, 68, 0.16);
        color: #f87171;
        border: 1px solid rgba(239, 68, 68, 0.35);
    }

    .pill-neutral {
        background-color: rgba(148, 163, 184, 0.16);
        color: #cbd5e1;
        border: 1px solid rgba(148, 163, 184, 0.30);
    }

    /* Recommendation / advisory cards */
    .rec-card {
        background-color: rgba(139, 92, 246, 0.08);
        border-left: 3px solid var(--accent-1);
        padding: 13px 16px;
        border-radius: 10px;
        margin-bottom: 10px;
        font-size: 14.5px;
        color: #dce3ec;
    }

    .rec-card.alert {
        background-color: rgba(239, 68, 68, 0.08);
        border-left: 3px solid #ef4444;
    }

    .rec-tag {
        font-size: 11px;
        letter-spacing: 1px;
        text-transform: uppercase;
        font-weight: 800;
        margin-right: 8px;
        color: var(--accent-1);
    }

    .rec-card.alert .rec-tag {
        color: #f87171;
    }

    .status-banner {
        border-radius: 12px;
        padding: 16px 20px;
        font-size: 15px;
        border: 1px solid rgba(255,255,255,0.08);
    }

    .status-banner.positive {
        background-color: rgba(34, 197, 94, 0.08);
        border-color: rgba(34, 197, 94, 0.3);
        color: #bbf7d0;
    }


    .group-card {
        background: linear-gradient(155deg, rgba(34, 184, 207, 0.08), rgba(255,255,255,0.015));
        border: 1px solid rgba(34, 184, 207, 0.18);
        border-radius: 14px;
        padding: 16px 18px;
        margin-bottom: 12px;
    }
    .group-card.variable {
        border-color: rgba(236, 72, 153, 0.18);
        background: linear-gradient(155deg, rgba(236, 72, 153, 0.07), rgba(255,255,255,0.015));
    }
    .group-title {
        font-size: 14px;
        font-weight: 800;
        letter-spacing: 1px;
        text-transform: uppercase;
        color: #eef1f7;
        margin-bottom: 5px;
    }
    .group-description {
        font-size: 12.5px;
        color: #9aa4b8;
        line-height: 1.55;
    }

    .status-banner.negative {
        background-color: rgba(239, 68, 68, 0.08);
        border-color: rgba(239, 68, 68, 0.3);
        color: #fecaca;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# STATEFUL, REALISM-CORRECTED APPLIANCE MODEL
#
# Two explicit operating groups are used in the live dashboard:
#
#   CONTINUOUS / 24H GROUP
#       refrigerator, washing_machine, stove, water_heater,
#       router_modem, radio, oven, microwave, deep_freezer, freezer
#
#   VARIABLE / TIME-DEPENDENT GROUP
#       air_conditioner, fans, lamps, phone_chargers, vacuum_cleaner
#
# Continuous devices remain active in parallel. Variable devices can
# turn on/off according to time and temperature and can overlap with
# one another and with the continuous group.
# ============================================================

APPLIANCES = {
    # -------------------- 24H / CONTINUOUS --------------------
    "refrigerator":     {"rated_kw": 0.15, "category": "continuous", "group": "24H"},
    "washing_machine":  {"rated_kw": 0.50, "category": "continuous", "group": "24H"},
    "stove":            {"rated_kw": 0.80, "category": "continuous", "group": "24H"},
    "water_heater":     {"rated_kw": 1.50, "category": "continuous", "group": "24H"},
    "router_modem":     {"rated_kw": 0.02, "category": "continuous", "group": "24H"},
    "radio":            {"rated_kw": 0.03, "category": "continuous", "group": "24H"},
    "oven":             {"rated_kw": 1.50, "category": "continuous", "group": "24H"},
    "microwave":        {"rated_kw": 1.00, "category": "continuous", "group": "24H"},
    "deep_freezer":     {"rated_kw": 0.20, "category": "continuous", "group": "24H"},
    "freezer":          {"rated_kw": 0.18, "category": "continuous", "group": "24H"},

    # -------------------- VARIABLE / TIME-DEPENDENT --------------------
    "air_conditioner": {"rated_kw": 1.20, "category": "variable", "group": "Variable"},
    "fans":             {"rated_kw": 0.075, "category": "variable", "group": "Variable"},
    "lamps":            {"rated_kw": 0.12, "category": "variable", "group": "Variable"},
    "phone_chargers":   {"rated_kw": 0.04, "category": "variable", "group": "Variable"},
    "vacuum_cleaner":   {"rated_kw": 0.70, "category": "variable", "group": "Variable"},
}


def _target_probability(name: str, hour: int, temperature: float) -> float:
    """Time/temperature dependent probability for variable appliances."""

    if name == "air_conditioner":
        if temperature >= 34:
            return 0.92
        if temperature >= 30:
            return 0.75
        if temperature >= 26:
            return 0.45
        if temperature >= 22:
            return 0.15
        return 0.03

    if name == "fans":
        if temperature >= 32:
            return 0.75
        if temperature >= 28:
            return 0.55
        if temperature >= 24:
            return 0.30
        return 0.05

    if name == "lamps":
        if 18 <= hour <= 23 or 0 <= hour < 6:
            return 0.85
        if 6 <= hour < 8:
            return 0.40
        return 0.05

    if name == "phone_chargers":
        if 18 <= hour <= 23:
            return 0.70
        if 7 <= hour <= 10:
            return 0.45
        return 0.12

    if name == "vacuum_cleaner":
        return 0.18 if 9 <= hour <= 17 else 0.01

    return 0.05


def _advance_appliance(name: str, config: dict, state: dict, hour: int, temperature: float) -> bool:
    """Advance one appliance state by a single live tick."""

    previous = state.get(name, {"on": False})

    # The 24H group is deliberately always active.
    if config["category"] == "continuous":
        is_on = True
    else:
        probability = _target_probability(name, hour, temperature)
        # Hysteresis prevents rapid flickering between Active/Standby.
        stay_on_probability = min(0.95, probability + 0.45)
        is_on = random.random() < (
            stay_on_probability if previous.get("on", False) else probability
        )

    state[name] = {"on": is_on}
    return is_on


def advance_and_generate(state: dict, hour: int, temperature: float, timestamp: datetime | None = None) -> dict:
    """Advance every appliance and return one complete live household reading."""

    timestamp = timestamp or datetime.now()
    appliance_data = {}
    total_power_kw = 0.0

    for name, config in APPLIANCES.items():
        is_on = _advance_appliance(name, config, state, hour, temperature)
        power = round(config["rated_kw"] * random.uniform(0.90, 1.10), 3) if is_on else 0.0

        appliance_data[name] = {
            "on": is_on,
            "power_kw": power,
            "group": config["group"],
            "operating_mode": "24-hour continuous" if config["group"] == "24H" else "time-dependent",
        }
        total_power_kw += power

    voltage = round(random.uniform(220, 240), 2)
    current = round((total_power_kw * 1000) / voltage, 2) if voltage > 0 else 0.0

    return {
        "timestamp": timestamp.isoformat(),
        "voltage_v": voltage,
        "current_a": current,
        "power_kw": round(total_power_kw, 3),
        "appliances": appliance_data,
    }


def _synthetic_hour_temperature(hour: int, anchor_temp: float, anchor_hour: int, swing: float = 6.0) -> float:
    """Diurnal temperature curve calibrated to the current live temperature."""

    def curve(h):
        return math.cos(2 * math.pi * (h - 15) / 24) * swing

    return anchor_temp + curve(hour) - curve(anchor_hour)


def backfill_history(inference_engine, appliance_state: dict, anchor_temp: float, anchor_hour: int, now=None):
    """Seed the inference engine with a synthetic week using the same appliance model."""

    now = now or datetime.now()

    for hours_ago in range(HISTORY_BACKFILL_HOURS, 0, -1):
        ts = now - timedelta(hours=hours_ago)
        temp = _synthetic_hour_temperature(ts.hour, anchor_temp, anchor_hour)
        reading = advance_and_generate(appliance_state, ts.hour, temp, timestamp=ts)
        inference_engine.add_measurement(reading["power_kw"])


def status_pill(text: str, kind: str = "neutral") -> str:
    return f'<span class="pill pill-{kind}">{text}</span>'


# ============================================================
# SHARED, PROCESS-WIDE STATE
#
# Everyone viewing this deployment — the main screen and any judge
# who scans the QR code — watches the SAME simulated household, not
# an independent random copy of it. That requires the engines, the
# appliance state machine, and the chart history all live in a single
# process-wide cache rather than per-browser session state.
# ============================================================

@st.cache_resource(show_spinner="Initializing the live energy system...")
def get_shared_state():
    return {"appliance_state": {}, "chart_history": []}


@st.cache_resource(show_spinner=False)
def get_realtime_engine():
    return RealtimeEngine(auto_location=False)


@st.cache_resource(show_spinner="Warming up the forecasting model with a week of history...")
def get_inference_engine(_shared, anchor_temp, anchor_hour):
    engine = InferenceEngine(model_path=str(FORECAST_MODEL_PATH))
    backfill_history(engine, _shared["appliance_state"], anchor_temp, anchor_hour)
    return engine


shared = get_shared_state()
realtime_engine = get_realtime_engine()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.markdown('<div class="eyebrow">EnergySavvy AI</div>', unsafe_allow_html=True)
st.sidebar.markdown("### System Overview")

st.sidebar.markdown(
    """
    **Live data pipeline**

    1. Location (selected below)
    2. Open-Meteo live weather
    3. Real-time household simulator
    4. Forecast model (trained on UCI + Cairo data)
    5. Anomaly detection
    6. Recommendation engine
    """
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Egyptian governorate**")

location_choice = st.sidebar.selectbox(
    "Select the demo governorate",
    options=list(EGYPT_GOVERNORATES.keys()) + ["Detect automatically (IP-based)"],
    index=0,
    label_visibility="collapsed",
)

if location_choice == "Detect automatically (IP-based)":
    st.sidebar.caption(
        "IP-based detection reflects the server's network location, which "
        "may differ from the presenter's physical location. Prefer a "
        "manual selection for live demonstrations."
    )
    detected = get_location_by_ip()
    active_lat = detected["latitude"]
    active_lon = detected["longitude"]
    active_city = detected["city"]
else:
    active_lat, active_lon = EGYPT_GOVERNORATES[location_choice]
    active_city = location_choice

realtime_engine.update_location(active_lat, active_lon, active_city)

st.sidebar.markdown("---")

st.sidebar.text_input(
    "Deployed application URL",
    key="app_url",
    placeholder="https://your-app.streamlit.app",
    help="Enter the public deployment URL to generate a QR code for mobile access.",
)

st.sidebar.markdown("---")

if HAS_AUTOREFRESH:
    st_autorefresh(interval=REFRESH_INTERVAL_SECONDS * 1000, key="live_refresh")
    st.sidebar.caption(f"Live refresh: every {REFRESH_INTERVAL_SECONDS} seconds • stable visual state")
else:
    st.sidebar.warning("Auto-refresh module not installed. Use manual refresh.")
    if st.sidebar.button("Refresh now"):
        st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown("### Team & Project")
st.sidebar.markdown(
    "**Team:** VoltAI  \n"
    "**Supervisor:** AbdelRahman Salem  \n"
    "**Team Leader & Member:** Mohamed Al-Saudi  \n"
    "**Location:** Cairo, Egypt"
)
st.sidebar.markdown(
    "**Project:** EnergySavvy AI  \n"
    "Software-based intelligent energy management system that analyzes household "
    "electricity consumption data to understand usage patterns, forecast future "
    "consumption, detect unusual behavior, and generate data-driven recommendations."
)
st.sidebar.markdown(
    "[LinkedIn](https://www.linkedin.com/in/mohamed-al-saudi-638274363/) &nbsp;•&nbsp; "
    "[GitHub](https://github.com/Mohamed-Al-Saudi) &nbsp;•&nbsp; "
    "[Project GitHub](https://github.com/Mohamed-Al-Saudi/EnergySavvy-AI-Project)"
)
st.sidebar.caption(
    "Electricity costs are estimated using the official EgyptERA "
    "residential tariff (2026 schedule)."
)
st.sidebar.caption("EnergySavvy AI  |  VoltAI  |  Cairo, Egypt")


# ============================================================
# PULL ONE LIVE TICK THROUGH THE FULL PIPELINE
# ============================================================

try:
    weather = get_current_weather(active_lat, active_lon)
except Exception as e:
    st.error("Unable to retrieve live weather data.")
    st.exception(e)
    st.stop()

now = datetime.now()

try:
    inference_engine = get_inference_engine(shared, weather["temperature_c"], now.hour)
except Exception as e:
    st.error("Unable to initialize the forecasting model.")
    st.exception(e)
    st.stop()

energy = advance_and_generate(shared["appliance_state"], now.hour, weather["temperature_c"], timestamp=now)

city = active_city
latitude = active_lat
longitude = active_lon

temperature = weather.get("temperature_c")
humidity = weather.get("humidity_percent")
wind_speed = weather.get("wind_speed_kmh")

power_kw = energy["power_kw"]
voltage_v = energy["voltage_v"]
current_a = energy["current_a"]
appliances = energy["appliances"]
reading_time = energy["timestamp"]

live_data = {
    "timestamp": reading_time,
    "weather": weather,
    "energy": energy,
    "location": {"city": city, "lat": latitude, "lon": longitude},
}

try:
    ai_results = inference_engine.process(live_data)
except Exception as e:
    st.error("The AI pipeline failed to process this reading.")
    st.exception(e)
    st.stop()

forecast = ai_results["forecast"]
anomaly = ai_results["anomaly"]
recommendations = ai_results["recommendations"]


# ============================================================
# FUTURE CONSUMPTION / BILL PROJECTION
# ============================================================

# Do NOT use the current instantaneous kW as a 24-hour baseline.
# The monthly estimate is derived from the operating behavior of all 15
# simulated appliances, including realistic duty cycles and time/temperature
# dependent usage.
bill_projection = estimate_monthly_usage(
    appliances,
    float(temperature or 30.0),
    now.hour,
)

projected_daily_kwh = bill_projection["daily_kwh"]
projected_monthly_kwh = bill_projection["monthly_kwh"]
projected_monthly_bill_egp = bill_projection["monthly_bill_egp"]
effective_rate = bill_projection["effective_rate"]
projected_daily_bill_egp = projected_monthly_bill_egp / 30.0
next_hour_cost_egp = (
    forecast["prediction_kw"] * effective_rate
    if forecast.get("available")
    else None
)


# ============================================================
# ROLLING SHARED CHART HISTORY
# ============================================================

shared["chart_history"].append(
    {
        "time": datetime.fromisoformat(reading_time),
        "power_kw": power_kw,
        "forecast_kw": forecast.get("prediction_kw"),
        "is_anomaly": anomaly.get("is_anomaly", False),
    }
)
shared["chart_history"] = shared["chart_history"][-CHART_HISTORY_POINTS:]

chart_df = pd.DataFrame(shared["chart_history"])


# ============================================================
# HEADER
# ============================================================

header_col, qr_col = st.columns([3, 1])

with header_col:
    st.markdown('<div class="eyebrow">Intelligent Energy Systems</div>', unsafe_allow_html=True)
    st.markdown('<div class="main-title">EnergySavvy AI</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">'
        '<span class="live-dot"></span>'
        "Live Egyptian household &nbsp;•&nbsp; real weather &nbsp;•&nbsp; simulated power &nbsp;•&nbsp; 3 AI pillars"
        f"<br><span style='font-size:12px;color:#8793aa;'>Active location: {city}</span>"
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="project-info">'
        '<strong>Project:</strong> EnergySavvy AI &nbsp;|&nbsp; '
        '<strong>Team:</strong> VoltAI &nbsp;|&nbsp; '
        '<strong>Supervisor:</strong> AbdelRahman Salem &nbsp;|&nbsp; '
        '<strong>Team Leader & Member:</strong> Mohamed Al-Saudi &nbsp;|&nbsp; Cairo, Egypt'
        '</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="pipeline-strip">'
        '<span class="pipeline-chip live">LIVE LOCATION</span>'
        '<span class="pipeline-chip weather">REAL WEATHER</span>'
        '<span class="pipeline-chip power">SIMULATED POWER</span>'
        '<span class="pipeline-chip ai">TRAINED AI</span>'
        '</div>',
        unsafe_allow_html=True,
    )

with qr_col:
    if HAS_QRCODE:
        app_url = st.session_state.get("app_url", "")
        if app_url:
            qr_img = qrcode.make(app_url)
            buf = io.BytesIO()
            qr_img.save(buf, format="PNG")
            st.image(buf.getvalue(), caption="Scan to open on a mobile device", width=110)


# ============================================================
# LIVE STATUS ROW (KPI cards)
# ============================================================

st.markdown('<div class="section-title">Live System Status</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-caption">Updated automatically from live location, weather, and household data</div>',
    unsafe_allow_html=True,
)

humidity_sub = f"{humidity:.0f}% relative humidity" if humidity is not None else "Humidity unavailable"
temperature_display = f"{float(temperature):.1f}\u00b0C" if temperature is not None else "N/A"

st.markdown(
    f"""
    <div class="kpi-row">
        <div class="glass-card">
            <div class="card-label">Live Location</div>
            <div class="card-value plain">{city}</div>
            <div class="card-sub">{latitude:.4f}, {longitude:.4f}</div>
        </div>
        <div class="glass-card">
            <div class="card-label">Real Weather</div>
            <div class="card-value plain">{temperature_display}</div>
            <div class="card-sub">{humidity_sub} &nbsp;|&nbsp; wind {wind_speed:.1f} km/h</div>
        </div>
        <div class="glass-card">
            <div class="card-label">Instantaneous Power</div>
            <div class="card-value">{power_kw:.3f} kW</div>
            <div class="card-sub">{voltage_v:.1f} V &nbsp;|&nbsp; {current_a:.2f} A &nbsp;|&nbsp; {active_count if 'active_count' in locals() else len([a for a in appliances.values() if a.get('on')])}/15 active</div>
        </div>
        <div class="glass-card">
            <div class="card-label">AI Pipeline</div>
            <div class="card-value plain">3 Pillars</div>
            <div class="card-sub">Forecast &nbsp;•&nbsp; Anomaly &nbsp;•&nbsp; Recommendations</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.expander("Environment details"):
    d1, d2 = st.columns(2)
    with d1:
        st.write("**City:**", city)
        st.write("**Latitude:**", latitude)
        st.write("**Wind speed:**", f"{wind_speed:.1f} km/h" if wind_speed is not None else "N/A")
    with d2:
        st.write("**Longitude:**", longitude)
        st.write("**Humidity:**", f"{humidity:.0f}%" if humidity is not None else "N/A")
        st.write("**Reading time:**", reading_time)


# ============================================================
# DASHBOARD DATA — LIVE CONSUMPTION HISTORY
# ============================================================

st.markdown('<div class="section-title">Live Consumption Dashboard</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-caption">Real weather and instantaneous household telemetry feeding the AI pipeline in real time.</div>',
    unsafe_allow_html=True,
)

fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=chart_df["time"],
        y=chart_df["power_kw"],
        mode="lines+markers",
        name="Live power (kW)",
        line=dict(color="#ff8c2b", width=3),
        fill="tozeroy",
        fillcolor="rgba(255,140,43,0.14)",
        marker=dict(
            size=7,
            color=["#ef4444" if a else "#ff8c2b" for a in chart_df["is_anomaly"]],
            line=dict(width=0),
        ),
    )
)

if chart_df["forecast_kw"].notna().any():
    fig.add_trace(
        go.Scatter(
            x=chart_df["time"],
            y=chart_df["forecast_kw"],
            mode="lines",
            name="AI forecast (kW)",
            line=dict(color="#22c55e", width=2, dash="dash"),
        )
    )

fig.update_layout(
    height=360,
    margin=dict(l=10, r=10, t=10, b=10),
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#cbd5e1"),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    xaxis=dict(showgrid=False),
    yaxis=dict(title="kW", showgrid=True, gridcolor="rgba(148,163,184,0.15)"),
)
st.plotly_chart(fig, use_container_width=True)


# ============================================================
# APPLIANCE DASHBOARD
# ============================================================

st.markdown('<div class="section-title">Household Device Dashboard</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-caption">The household is explicitly separated into 24-hour continuous devices and time-dependent devices. Both groups may operate in parallel.</div>',
    unsafe_allow_html=True,
)

appliance_rows = [
    {
        "Appliance": name.replace("_", " ").title(),
        "Operating Group": data.get("group", "Variable"),
        "Mode": data.get("operating_mode", "time-dependent"),
        "Status": "Active" if data.get("on") else "Standby",
        "Power (kW)": data.get("power_kw", 0.0),
    }
    for name, data in appliances.items()
]
appliance_df = pd.DataFrame(appliance_rows)
continuous_df = appliance_df[appliance_df["Operating Group"] == "24H"].copy()
variable_df = appliance_df[appliance_df["Operating Group"] == "Variable"].copy()
active_count = int((appliance_df["Status"] == "Active").sum())
continuous_power = float(continuous_df["Power (kW)"].sum())
variable_power = float(variable_df["Power (kW)"].sum())

c1, c2, c3 = st.columns(3)
with c1:
    st.markdown(
        f"""<div class="group-card"><div class="group-title">24-hour continuous</div><div class="group-description">{len(continuous_df)} devices · active in parallel · instantaneous load {continuous_power:.3f} kW</div></div>""",
        unsafe_allow_html=True,
    )
with c2:
    st.markdown(
        f"""<div class="group-card variable"><div class="group-title">Time-dependent</div><div class="group-description">{len(variable_df)} devices · controlled by time/temperature · instantaneous load {variable_power:.3f} kW</div></div>""",
        unsafe_allow_html=True,
    )
with c3:
    st.markdown(
        f"""<div class="group-card"><div class="group-title">Live household</div><div class="group-description">{active_count} / {len(appliance_df)} active · total instantaneous load {power_kw:.3f} kW</div></div>""",
        unsafe_allow_html=True,
    )

left_devices, right_devices = st.columns(2)
with left_devices:
    st.markdown('<div class="card-label">24-hour continuous devices</div>', unsafe_allow_html=True)
    st.dataframe(
        continuous_df[["Appliance", "Status", "Power (kW)"]],
        use_container_width=True,
        hide_index=True,
    )
with right_devices:
    st.markdown('<div class="card-label">Time-dependent devices</div>', unsafe_allow_html=True)
    st.dataframe(
        variable_df[["Appliance", "Status", "Power (kW)"]],
        use_container_width=True,
        hide_index=True,
    )

active_df = appliance_df[appliance_df["Power (kW)"] > 0].sort_values("Power (kW)", ascending=True)
if not active_df.empty:
    bar_fig = go.Figure(
        go.Bar(
            x=active_df["Power (kW)"],
            y=active_df["Appliance"],
            orientation="h",
            marker=dict(
                color=active_df["Power (kW)"],
                colorscale=[[0, "#22b8cf"], [0.5, "#8b5cf6"], [1, "#ec4899"]],
            ),
            customdata=active_df[["Operating Group"]],
            hovertemplate="%{y}<br>Power: %{x:.3f} kW<br>Group: %{customdata[0]}<extra></extra>",
        )
    )
    bar_fig.update_layout(
        height=400,
        margin=dict(l=10, r=10, t=10, b=10),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#cbd5e1"),
        xaxis=dict(title="Instantaneous kW", gridcolor="rgba(148,163,184,0.15)"),
        yaxis=dict(title="", showgrid=False),
    )
    st.plotly_chart(bar_fig, use_container_width=True)


# ============================================================
# 1. FORECAST + FUTURE MONTHLY BILL
# ============================================================

st.markdown('<div class="section-title">1. Forecast & Future Monthly Bill</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-caption">The trained Random Forest forecasts the next hour, while the monthly bill is estimated from appliance operating schedules rather than multiplying one instantaneous reading by 720 hours.</div>',
    unsafe_allow_html=True,
)

forecast_col, bill_col = st.columns([1, 1])

with forecast_col:
    if forecast.get("available"):
        forecast_value = f"{forecast['prediction_kw']:.3f} kW"
        forecast_cost = (
            f"Estimated next-hour energy cost: {next_hour_cost_egp:.2f} EGP"
            if next_hour_cost_egp is not None
            else ""
        )
        forecast_status = status_pill("MODEL ACTIVE", "positive")
    else:
        forecast_value = "Initializing"
        forecast_cost = forecast.get("message", "Building the forecast history...")
        forecast_status = status_pill("WARMING UP", "neutral")

    st.markdown(
        f"""
        <div class="glass-card">
            <div class="card-label">Next-Hour Consumption Forecast</div>
            <div class="card-value">{forecast_value}</div>
            <div class="card-sub">{forecast_cost}</div>
            <div style="margin-top:14px;">{forecast_status}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with bill_col:
    st.markdown(
        f"""
        <div class="glass-card bill-card">
            <div class="card-label">Projected Monthly Electricity Bill</div>
            <div class="card-value">{projected_monthly_bill_egp:,.0f} EGP</div>
            <div class="card-sub">
                Estimated consumption: <strong>{projected_monthly_kwh:,.0f} kWh/month</strong><br>
                Average modeled use: <strong>{projected_daily_kwh:.1f} kWh/day</strong><br>
                Effective modeled cost: <strong>{effective_rate:.2f} EGP/kWh</strong>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<div style='height:14px;'></div>", unsafe_allow_html=True)

bill_note_col1, bill_note_col2 = st.columns([2, 1])
with bill_note_col1:
    st.markdown(
        f"""
        <div class="status-banner positive">
            <strong>Why this estimate is more realistic:</strong>
            the simulator's 24H appliances remain available in parallel, but their
            billing contribution uses duty-cycle behavior. Variable appliances use
            time and temperature dependent operating probabilities. The model therefore
            does not assume the current {power_kw:.3f} kW load continues unchanged for all 30 days.
        </div>
        """,
        unsafe_allow_html=True,
    )
with bill_note_col2:
    st.markdown(
        f"""
        <div class="glass-card">
            <div class="card-label">Daily Cost Projection</div>
            <div class="card-value plain" style="font-size:24px;">{projected_daily_bill_egp:.2f} EGP</div>
            <div class="card-sub">30-day projection</div>
        </div>
        """,
        unsafe_allow_html=True,
    )




# ============================================================
# 2. UNUSUAL BEHAVIOR / ANOMALY DETECTION
# ============================================================

st.markdown('<div class="section-title">2. Detect Unusual Behavior</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-caption">The live anomaly layer compares the current household load with its recent learned baseline and flags statistically unusual behavior.</div>',
    unsafe_allow_html=True,
)

if anomaly.get("available"):
    z_score = anomaly["score"]
    gauge_fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=z_score,
            number={"suffix": " z", "font": {"color": "#eef1f7"}},
            gauge={
                "axis": {"range": [-4, 4], "tickcolor": "#8b96ab"},
                "bar": {"color": "#ef4444" if anomaly.get("is_anomaly") else "#8b5cf6"},
                "steps": [
                    {"range": [-4, -3], "color": "rgba(239,68,68,0.35)"},
                    {"range": [-3, 3], "color": "rgba(139,92,246,0.18)"},
                    {"range": [3, 4], "color": "rgba(239,68,68,0.35)"},
                ],
                "threshold": {
                    "line": {"color": "#f87171", "width": 3},
                    "thickness": 0.8,
                    "value": z_score,
                },
                "bgcolor": "rgba(0,0,0,0)",
            },
        )
    )
    gauge_fig.update_layout(
        height=220,
        margin=dict(l=20, r=20, t=20, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#eef1f7"),
    )

    ac1, ac2 = st.columns([1.2, 1])
    with ac1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-label">Anomaly Score</div>', unsafe_allow_html=True)
        st.plotly_chart(gauge_fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with ac2:
        if anomaly.get("is_anomaly"):
            st.markdown(
                f"""<div class="status-banner negative"><strong>Unusual behavior detected.</strong><br><br>Baseline: {anomaly['baseline_kw']:.3f} kW<br>Current: {power_kw:.3f} kW<br>Deviation score: {z_score:.2f} z</div>""",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""<div class="status-banner positive"><strong>Consumption pattern is currently normal.</strong><br><br>Baseline: {anomaly['baseline_kw']:.3f} kW<br>Current: {power_kw:.3f} kW<br>Deviation score: {z_score:.2f} z</div>""",
                unsafe_allow_html=True,
            )
else:
    st.markdown(
        f"""<div class="glass-card"><div class="card-label">Anomaly Detection</div><div class="card-value plain" style="font-size:18px;">{anomaly.get('message', 'Initializing anomaly baseline')}</div></div>""",
        unsafe_allow_html=True,
    )


# ============================================================
# 3. DATA-DRIVEN RECOMMENDATIONS
# ============================================================

st.markdown('<div class="section-title">3. Data-Driven Recommendations</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-caption">Recommendations combine the live appliance states, current consumption, time of day, temperature, and detected behavior.</div>',
    unsafe_allow_html=True,
)

if recommendations:
    for rec in recommendations:
        is_alert = "unusual" in rec.lower() or "anomaly" in rec.lower()
        css_class = "rec-card alert" if is_alert else "rec-card"
        tag = "Alert" if is_alert else "Recommendation"
        st.markdown(
            f'<div class="{css_class}"><span class="rec-tag">{tag}</span>{rec}</div>',
            unsafe_allow_html=True,
        )
else:
    st.markdown(
        '<div class="rec-card"><span class="rec-tag">Status</span>No immediate energy-saving action is required.</div>',
        unsafe_allow_html=True,
    )


# ============================================================
# FINAL SNAPSHOT
# ============================================================

st.markdown('<div class="section-title">System Snapshot</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-caption">The complete live state presented to the judge at the current refresh.</div>',
    unsafe_allow_html=True,
)

summary_df = pd.DataFrame(
    [
        {"Metric": "Location", "Value": city},
        {"Metric": "Live temperature", "Value": f"{temperature:.1f} °C" if temperature is not None else "N/A"},
        {"Metric": "Humidity", "Value": f"{humidity:.0f}%" if humidity is not None else "N/A"},
        {"Metric": "Instantaneous power", "Value": f"{power_kw:.3f} kW"},
        {"Metric": "Voltage", "Value": f"{voltage_v:.1f} V"},
        {"Metric": "Current", "Value": f"{current_a:.2f} A"},
        {"Metric": "Forecast next hour", "Value": f"{forecast['prediction_kw']:.3f} kW" if forecast.get("available") else "Initializing"},
        {"Metric": "Projected monthly consumption", "Value": f"{projected_monthly_kwh:,.0f} kWh"},
        {"Metric": "Projected monthly bill", "Value": f"{projected_monthly_bill_egp:,.0f} EGP"},
        {"Metric": "Anomaly status", "Value": "Anomaly Detected" if anomaly.get("is_anomaly") else "Normal"},
        {"Metric": "Active appliances", "Value": f"{active_count} of {len(appliance_df)}"},
    ]
)
st.dataframe(summary_df, use_container_width=True, hide_index=True)


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")
st.markdown(
    '<div class="footer-card">'
    '<div class="footer-title">EnergySavvy AI &nbsp;|&nbsp; VoltAI</div>'
    '<div class="footer-line"><strong>Project:</strong> EnergySavvy AI &nbsp;•&nbsp; '
    '<strong>Supervisor:</strong> AbdelRahman Salem &nbsp;•&nbsp; '
    '<strong>Team Leader & Member:</strong> Mohamed Al-Saudi &nbsp;•&nbsp; Cairo, Egypt</div>'
    '<div class="footer-line">'
    '<a href="mailto:mohamed.alsuadi2007@gmail.com">mohamed.alsuadi2007@gmail.com</a> &nbsp;•&nbsp; '
    '<a href="https://www.linkedin.com/in/mohamed-al-saudi-638274363/" target="_blank">LinkedIn</a> &nbsp;•&nbsp; '
    '<a href="https://github.com/Mohamed-Al-Saudi" target="_blank">GitHub</a> &nbsp;•&nbsp; '
    '<a href="https://github.com/Mohamed-Al-Saudi/EnergySavvy-AI-Project" target="_blank">Project GitHub</a>'
    '</div>'
    '<div class="footer-description">'
    'EnergySavvy AI is a software-based intelligent energy management system that analyzes household electricity consumption data to understand usage patterns, forecast future consumption, detect unusual behavior, and generate data-driven recommendations.'
    '</div>'
    '</div>',
    unsafe_allow_html=True,
)
st.caption(
    "The forecast model was trained on the UCI Household Power Consumption dataset "
    "and the Cairo Weather dataset (historical, training-only). All data displayed "
    "above is generated live: real, presenter-selected location, real Open-Meteo "
    "weather, and a live household simulator feeding the trained model. Electricity "
    "costs are estimated using the EgyptERA 2026 residential tariff and are "
    "provided for illustration only."
)
st.caption(f"Last update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
