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
EGYPT_CITIES = {
    "Cairo": (30.0444, 31.2357),
    "Giza": (30.0131, 31.2089),
    "Alexandria": (31.2001, 29.9187),
    "Mansoura": (31.0409, 31.3785),
    "Tanta": (30.7865, 31.0004),
    "Ismailia": (30.5965, 32.2715),
    "Port Said": (31.2653, 32.3019),
    "Luxor": (25.6872, 32.6396),
    "Aswan": (24.0889, 32.8998),
    "Sharm El Sheikh": (27.9158, 34.3300),
}

# Official EgyptERA residential tariff (2026), cumulative tiers.
# Source: Egyptian Electricity Holding Company / EgyptERA published
# rates. Provided for illustration; actual bills may include
# additional fixed fees and should be verified against the latest
# official schedule.
TARIFF_TIERS_EGP_PER_KWH = [
    (50, 0.68),
    (100, 0.95),
    (200, 1.15),
    (350, 1.72),
    (650, 2.18),
    (1000, 2.40),
    (float("inf"), 2.58),
]


def calculate_bill_egp(monthly_kwh: float) -> float:
    """Cumulative tiered bill calculation using the EgyptERA schedule."""

    if monthly_kwh <= 0:
        return 0.0

    remaining = monthly_kwh
    cost = 0.0
    previous_cap = 0

    for cap, rate in TARIFF_TIERS_EGP_PER_KWH:
        band_width = remaining if cap == float("inf") else min(remaining, cap - previous_cap)
        if band_width <= 0:
            break
        cost += band_width * rate
        remaining -= band_width
        previous_cap = cap
        if remaining <= 0:
            break

    return cost


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

    .stApp {
        background-color: #0b0b16;
        background-image:
            radial-gradient(at 15% 20%, rgba(139, 92, 246, 0.22) 0px, transparent 55%),
            radial-gradient(at 85% 15%, rgba(236, 72, 153, 0.18) 0px, transparent 55%),
            radial-gradient(at 50% 90%, rgba(34, 184, 207, 0.14) 0px, transparent 55%);
        background-attachment: fixed;
        background-size: 200% 200%;
        animation: bgDrift 24s ease-in-out infinite alternate;
    }

    @keyframes bgDrift {
        0%   { background-position: 0% 0%, 100% 0%, 50% 100%; }
        100% { background-position: 15% 15%, 85% 20%, 55% 85%; }
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
st.sidebar.markdown("**Household location**")

location_choice = st.sidebar.selectbox(
    "Select the demo location",
    options=list(EGYPT_CITIES.keys()) + ["Detect automatically (IP-based)"],
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
    active_lat, active_lon = EGYPT_CITIES[location_choice]
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
    st.sidebar.caption(f"Auto-refreshing every {REFRESH_INTERVAL_SECONDS} seconds")
else:
    st.sidebar.warning("Auto-refresh module not installed. Use manual refresh.")
    if st.sidebar.button("Refresh now"):
        st.rerun()

st.sidebar.markdown("---")
st.sidebar.caption(
    "Electricity costs are estimated using the official EgyptERA "
    "residential tariff (2026 schedule)."
)
st.sidebar.caption("EnergySavvy AI  |  Intelligent Systems")


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
# COST-OF-BILL PROJECTION
# ============================================================

baseline_kw = anomaly.get("baseline_kw") if anomaly.get("available") else power_kw
projected_daily_kwh = baseline_kw * 24
projected_monthly_kwh = projected_daily_kwh * 30
projected_monthly_bill_egp = calculate_bill_egp(projected_monthly_kwh)
effective_rate = (
    projected_monthly_bill_egp / projected_monthly_kwh
    if projected_monthly_kwh > 0
    else TARIFF_TIERS_EGP_PER_KWH[0][1]
)
projected_daily_bill_egp = projected_daily_kwh * effective_rate
next_hour_cost_egp = (
    forecast["prediction_kw"] * effective_rate if forecast.get("available") else None
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
        "Live forecasting, anomaly detection, and cost-aware recommendations "
        f"for a household in {city}"
        "</div>",
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
            <div class="card-label">Location</div>
            <div class="card-value plain">{city}</div>
            <div class="card-sub">Live weather anchored to this location</div>
        </div>
        <div class="glass-card">
            <div class="card-label">Temperature</div>
            <div class="card-value plain">{temperature_display}</div>
            <div class="card-sub">{humidity_sub} &nbsp;|&nbsp; wind {wind_speed:.1f} km/h</div>
        </div>
        <div class="glass-card">
            <div class="card-label">Current Power Draw</div>
            <div class="card-value">{power_kw:.3f} kW</div>
            <div class="card-sub">{voltage_v:.1f} V &nbsp;|&nbsp; {current_a:.2f} A</div>
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
# AI INSIGHTS — the three core deliverables, front and center:
# forecast + cost, anomaly detection, recommendations.
# ============================================================

st.markdown('<div class="section-title">AI Insights</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-caption">Forecast, anomaly detection, and recommendations produced by the trained model from the live feed</div>',
    unsafe_allow_html=True,
)

insight_col1, insight_col2 = st.columns([1, 1])

with insight_col1:
    if forecast.get("available"):
        forecast_value_html = f"{forecast['prediction_kw']:.3f} kW"
        cost_line = f"Estimated cost: {next_hour_cost_egp:.2f} EGP for the next hour" if next_hour_cost_egp is not None else ""
    else:
        forecast_value_html = "Initializing"
        cost_line = forecast.get("message", "")

    st.markdown(
        f"""
        <div class="glass-card">
            <div class="card-label">Forecast — Next Hour</div>
            <div class="card-value">{forecast_value_html}</div>
            <div class="card-sub">{cost_line}</div>
            <hr style="border-color: rgba(255,255,255,0.08); margin: 14px 0;">
            <div class="card-label">Projected Monthly Bill</div>
            <div class="card-value plain" style="font-size: 22px;">{projected_monthly_bill_egp:,.0f} EGP</div>
            <div class="card-sub">
                Based on the current 24-hour average of {baseline_kw:.3f} kW,
                projected to {projected_monthly_kwh:,.0f} kWh/month at the
                EgyptERA residential tariff. Daily estimate:
                {projected_daily_bill_egp:,.2f} EGP.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with insight_col2:
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
            height=200,
            margin=dict(l=20, r=20, t=30, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#eef1f7"),
        )

        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-label">Anomaly Detection</div>', unsafe_allow_html=True)
        st.plotly_chart(gauge_fig, use_container_width=True)
        if anomaly.get("is_anomaly"):
            st.markdown(
                f'{status_pill("Anomaly Detected", "negative")} '
                f'<span class="card-sub">Baseline {anomaly["baseline_kw"]:.3f} kW, current {power_kw:.3f} kW</span>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'{status_pill("Normal", "positive")} '
                f'<span class="card-sub">Baseline {anomaly["baseline_kw"]:.3f} kW, current {power_kw:.3f} kW</span>',
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.markdown(
            f"""
            <div class="glass-card">
                <div class="card-label">Anomaly Detection</div>
                <div class="card-value plain" style="font-size: 18px;">{anomaly.get("message")}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

st.markdown('<div class="card-label" style="padding-left: 4px;">Recommendations</div>', unsafe_allow_html=True)

if recommendations:
    for rec in recommendations:
        is_alert = "unusual" in rec.lower()
        css_class = "rec-card alert" if is_alert else "rec-card"
        tag = "Alert" if is_alert else "Recommendation"
        st.markdown(
            f'<div class="{css_class}"><span class="rec-tag">{tag}</span>{rec}</div>',
            unsafe_allow_html=True,
        )
else:
    st.markdown(
        '<div class="rec-card"><span class="rec-tag">Status</span>'
        "No action is required at this time.</div>",
        unsafe_allow_html=True,
    )


# ============================================================
# CONSUMPTION CHART: ACTUAL vs FORECAST
# ============================================================

st.markdown('<div class="section-title">Consumption: Live vs. Forecast</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-caption">Highlighted markers indicate readings flagged as anomalous</div>',
    unsafe_allow_html=True,
)

fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=chart_df["time"],
        y=chart_df["power_kw"],
        mode="lines+markers",
        name="Live power (kW)",
        line=dict(color="#8b5cf6", width=3),
        fill="tozeroy",
        fillcolor="rgba(139, 92, 246, 0.15)",
        marker=dict(
            size=8,
            color=["#ef4444" if a else "#ec4899" for a in chart_df["is_anomaly"]],
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
            name="Model forecast (kW)",
            line=dict(color="#f97316", width=2, dash="dash"),
        )
    )

fig.update_layout(
    height=380,
    margin=dict(l=10, r=10, t=10, b=10),
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#cbd5e1"),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    xaxis=dict(showgrid=False),
    yaxis=dict(title="kW", showgrid=True, gridcolor="rgba(148,163,184,0.15)"),
)

st.plotly_chart(fig, use_container_width=True)

if not forecast.get("available"):
    st.info(forecast.get("message"))


# ============================================================
# APPLIANCE BREAKDOWN
# ============================================================

st.markdown('<div class="section-title">Appliance Breakdown</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-caption">The simulator explicitly separates continuous 24-hour loads from flexible time-dependent loads. Both groups can operate in parallel.</div>',
    unsafe_allow_html=True,
)

appliance_rows = [
    {
        "Appliance": name.replace("_", " " ).title(),
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
        f"""<div class=\"group-card\"><div class=\"group-title\">24-hour continuous</div><div class=\"group-description\">{len(continuous_df)} appliances · always active · can operate in parallel · current load {continuous_power:.3f} kW</div></div>""",
        unsafe_allow_html=True,
    )
with c2:
    st.markdown(
        f"""<div class=\"group-card variable\"><div class=\"group-title\">Time-dependent</div><div class=\"group-description\">{len(variable_df)} appliances · may turn on/off by time and temperature · can overlap with each other and the 24H group · current load {variable_power:.3f} kW</div></div>""",
        unsafe_allow_html=True,
    )
with c3:
    st.markdown(
        f"""<div class=\"group-card\"><div class=\"group-title\">Live household</div><div class=\"group-description\">{active_count} of {len(appliance_df)} appliances active · total instantaneous load {power_kw:.3f} kW</div></div>""",
        unsafe_allow_html=True,
    )

st.markdown('<div class="card-label" style="padding-left: 4px;">24-hour continuous devices</div>', unsafe_allow_html=True)
st.dataframe(continuous_df[["Appliance", "Status", "Power (kW)"]], use_container_width=True, hide_index=True)

st.markdown('<div class="card-label" style="padding-left: 4px; margin-top: 12px;">Time-dependent devices</div>', unsafe_allow_html=True)
st.dataframe(variable_df[["Appliance", "Status", "Power (kW)"]], use_container_width=True, hide_index=True)

st.markdown('<div class="card-label" style="padding-left: 4px; margin-top: 12px;">Current load contribution</div>', unsafe_allow_html=True)
active_df = appliance_df[appliance_df["Power (kW)"] > 0].sort_values("Power (kW)", ascending=True)
if not active_df.empty:
    bar_fig = go.Figure(go.Bar(
        x=active_df["Power (kW)"],
        y=active_df["Appliance"],
        orientation="h",
        marker=dict(color=active_df["Power (kW)"], colorscale=[[0, "#22b8cf"], [0.5, "#8b5cf6"], [1, "#ec4899"]]),
        customdata=active_df[["Operating Group"]],
        hovertemplate="%{y}<br>Power: %{x:.3f} kW<br>Group: %{customdata[0]}<extra></extra>",
    ))
    bar_fig.update_layout(
        height=430,
        margin=dict(l=10, r=10, t=10, b=10),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#cbd5e1"),
        xaxis=dict(title="Instantaneous kW", gridcolor="rgba(148,163,184,0.15)"),
        yaxis=dict(title="", showgrid=False),
    )
    st.plotly_chart(bar_fig, use_container_width=True)
else:
    st.info("No appliances are currently drawing power.")


# ============================================================
# SYSTEM SUMMARY
# ============================================================

st.markdown('<div class="section-title">Summary</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-caption">Consolidated snapshot of the current reading</div>',
    unsafe_allow_html=True,
)

summary_df = pd.DataFrame(
    [
        {"Metric": "Location", "Value": city},
        {"Metric": "Temperature", "Value": f"{temperature:.1f} \u00b0C" if temperature is not None else "N/A"},
        {"Metric": "Current Power", "Value": f"{power_kw:.3f} kW"},
        {"Metric": "Voltage", "Value": f"{voltage_v:.1f} V"},
        {"Metric": "Current", "Value": f"{current_a:.2f} A"},
        {
            "Metric": "Forecast (next hour)",
            "Value": f"{forecast['prediction_kw']:.3f} kW" if forecast.get("available") else "Initializing",
        },
        {
            "Metric": "Estimated next-hour cost",
            "Value": f"{next_hour_cost_egp:.2f} EGP" if next_hour_cost_egp is not None else "N/A",
        },
        {"Metric": "Projected monthly bill", "Value": f"{projected_monthly_bill_egp:,.0f} EGP"},
        {
            "Metric": "Anomaly Status",
            "Value": "Anomaly Detected" if anomaly.get("is_anomaly") else "Normal",
        },
        {"Metric": "Active Appliances", "Value": f"{active_count} of {len(appliance_df)}"},
    ]
)

st.dataframe(summary_df, use_container_width=True, hide_index=True)


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")
st.caption("EnergySavvy AI  |  Intelligent Systems")
st.caption(
    "The forecast model was trained on the UCI Household Power Consumption dataset "
    "and the Cairo Weather dataset (historical, training-only). All data displayed "
    "above is generated live: real, presenter-selected location, real Open-Meteo "
    "weather, and a live household simulator feeding the trained model. Electricity "
    "costs are estimated using the EgyptERA 2026 residential tariff and are "
    "provided for illustration only."
)
st.caption(f"Last update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
