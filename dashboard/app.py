"""
EnergySavvy AI
---------------
Final Streamlit deployment application.

Pipeline:
    Live Location (ipapi.co)
        -> Open-Meteo Weather
        -> RealtimeEngine
        -> EnergySimulator
        -> Forecast Model (forecast_rf.pkl)
        -> Anomaly Detection
        -> Recommendations
        -> Streamlit Dashboard

IMPORTANT:
    The UCI Household Power Consumption dataset and the Cairo Weather
    dataset were used ONLY to train forecast_rf.pkl (see notebooks/).
    Nothing in this file reads those historical CSVs. Everything shown
    here is generated live: real weather, real location, and a live
    household simulator whose output is fed straight into the trained
    model.

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
# dashboard/app.py lives at <project_root>/dashboard/app.py, so the
# project root is one level up. We add it to sys.path so `src.*`
# imports work no matter what directory `streamlit run` is launched
# from.

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

# Optional auto-refresh. The app still works without it (a manual
# refresh button appears instead), so a missing dependency never
# takes the dashboard down during a live demonstration.
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


# ============================================================
# STREAMLIT PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="EnergySavvy AI",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# STYLE
# ============================================================
# A restrained, formal palette: deep slate background tones, a single
# cyan accent for emphasis, muted status colors, and no iconography
# beyond simple geometric indicators (dots, pills, bars).

st.markdown(
    """
    <style>

    html, body, [class*="css"] {
        font-family: "Segoe UI", "Inter", "Helvetica Neue", Arial, sans-serif;
    }

    .eyebrow {
        font-size: 12px;
        letter-spacing: 3px;
        text-transform: uppercase;
        color: #7c8798;
        font-weight: 600;
        margin-bottom: 4px;
    }

    .main-title {
        font-size: 40px;
        font-weight: 700;
        letter-spacing: -0.5px;
        margin-bottom: 4px;
        color: #eef2f7;
    }

    .subtitle {
        font-size: 15px;
        color: #9aa4b2;
        margin-bottom: 22px;
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
        font-size: 20px;
        font-weight: 700;
        color: #e6eaf0;
        margin-top: 34px;
        margin-bottom: 4px;
        border-left: 3px solid #22b8cf;
        padding-left: 12px;
    }

    .section-caption {
        font-size: 13px;
        color: #7c8798;
        margin-bottom: 14px;
        padding-left: 15px;
    }

    /* KPI cards */
    .kpi-row {
        display: flex;
        gap: 16px;
        flex-wrap: wrap;
        margin-bottom: 6px;
    }

    .kpi-card {
        flex: 1;
        min-width: 190px;
        background: linear-gradient(155deg, rgba(255,255,255,0.045), rgba(255,255,255,0.015));
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 12px;
        padding: 18px 20px;
        transition: border-color 0.3s ease;
    }

    .kpi-card:hover {
        border-color: rgba(34, 184, 207, 0.45);
    }

    .kpi-label {
        font-size: 12px;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        color: #7c8798;
        font-weight: 600;
        margin-bottom: 8px;
    }

    .kpi-value {
        font-size: 28px;
        font-weight: 700;
        color: #eef2f7;
        line-height: 1.1;
    }

    .kpi-sub {
        font-size: 12.5px;
        color: #8b96a5;
        margin-top: 6px;
    }

    /* Status pills */
    .pill {
        display: inline-block;
        padding: 3px 12px;
        border-radius: 999px;
        font-size: 12.5px;
        font-weight: 600;
        letter-spacing: 0.3px;
    }

    .pill-positive {
        background-color: rgba(34, 197, 94, 0.14);
        color: #4ade80;
        border: 1px solid rgba(34, 197, 94, 0.35);
    }

    .pill-negative {
        background-color: rgba(239, 68, 68, 0.14);
        color: #f87171;
        border: 1px solid rgba(239, 68, 68, 0.35);
    }

    .pill-neutral {
        background-color: rgba(148, 163, 184, 0.14);
        color: #cbd5e1;
        border: 1px solid rgba(148, 163, 184, 0.30);
    }

    /* Recommendation / advisory cards */
    .rec-card {
        background-color: rgba(34, 184, 207, 0.07);
        border-left: 3px solid #22b8cf;
        padding: 13px 16px;
        border-radius: 8px;
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
        font-weight: 700;
        margin-right: 8px;
        color: #22b8cf;
    }

    .rec-card.alert .rec-tag {
        color: #f87171;
    }

    .status-banner {
        border-radius: 10px;
        padding: 16px 20px;
        font-size: 15px;
        border: 1px solid rgba(255,255,255,0.08);
    }

    .status-banner.positive {
        background-color: rgba(34, 197, 94, 0.08);
        border-color: rgba(34, 197, 94, 0.3);
        color: #bbf7d0;
    }

    .status-banner.negative {
        background-color: rgba(239, 68, 68, 0.08);
        border-color: rgba(239, 68, 68, 0.3);
        color: #fecaca;
    }

    .status-banner.neutral {
        background-color: rgba(148, 163, 184, 0.08);
        color: #cbd5e1;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


def status_pill(text: str, kind: str = "neutral") -> str:
    """Return an inline HTML status badge (kind: positive, negative, neutral)."""
    return f'<span class="pill pill-{kind}">{text}</span>'


# ============================================================
# SIMULATED-HISTORY BACKFILL
#
# The trained forecast model requires 168 chronological hourly power
# readings (lag_1 through lag_168, plus rolling statistics) before it
# can produce a prediction, and 24 readings before anomaly detection
# has a baseline. Rather than requiring a full week of wall-clock time
# before any output is available, the application seeds the
# InferenceEngine's history with 168 synthetic hourly points generated
# using the same equation-based logic as EnergySimulator (appliance
# probability curves keyed by hour-of-day and temperature). A diurnal
# temperature curve, anchored to the current live temperature, drives
# realistic hour-to-hour variation without requiring a historical
# weather API call.
# ============================================================

def _synthetic_hour_temperature(
    hour: int,
    anchor_temp: float,
    anchor_hour: int,
    swing: float = 6.0,
) -> float:
    """Diurnal temperature curve, calibrated so the temperature at
    anchor_hour equals anchor_temp exactly (peak near 15:00, trough near 03:00)."""

    def curve(h):
        return math.cos(2 * math.pi * (h - 15) / 24) * swing

    return anchor_temp + curve(hour) - curve(anchor_hour)


def _simulate_power_for_hour(simulator, hour: int, temperature: float) -> float:
    """Reproduce EnergySimulator's own probability model for an
    arbitrary historical hour. EnergySimulator.generate() only ever
    uses datetime.now(), so this reuses its appliance table and
    probability function directly for backfill purposes."""

    total_power_kw = 0.0

    for name, config in simulator.appliances.items():
        probability = simulator._get_probability(name, hour, temperature)

        if random.random() < probability:
            total_power_kw += config["rated_power_kw"] * random.uniform(0.90, 1.10)

    return round(total_power_kw, 3)


def backfill_history(inference_engine, simulator, anchor_temp, anchor_hour, now=None):
    """Seed inference_engine.power_history with a synthetic week so
    forecasting and anomaly detection are available immediately."""

    now = now or datetime.now()

    for hours_ago in range(HISTORY_BACKFILL_HOURS, 0, -1):
        ts = now - timedelta(hours=hours_ago)
        temp = _synthetic_hour_temperature(ts.hour, anchor_temp, anchor_hour)
        power = _simulate_power_for_hour(simulator, ts.hour, temp)
        inference_engine.add_measurement(power)


# ============================================================
# CACHED ENGINE INITIALIZATION
# (created once per server process, shared across reruns)
# ============================================================

@st.cache_resource(show_spinner="Resolving location and connecting to live weather data...")
def get_realtime_engine():
    return RealtimeEngine(auto_location=True)


@st.cache_resource(show_spinner="Initializing the forecasting model with historical context...")
def get_inference_engine(_realtime_engine):
    engine = InferenceEngine(model_path=str(FORECAST_MODEL_PATH))

    seed_live = _realtime_engine.get_live_data()
    seed_temp = seed_live["weather"]["temperature_c"]
    seed_hour = datetime.now().hour

    backfill_history(engine, _realtime_engine.simulator, seed_temp, seed_hour)

    return engine


try:
    realtime_engine = get_realtime_engine()
    inference_engine = get_inference_engine(realtime_engine)
except Exception as e:
    st.error("Unable to initialize the live energy system.")
    st.exception(e)
    st.stop()


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
        "Live forecasting, anomaly detection, and recommendations for real Egyptian households"
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
# SIDEBAR
# ============================================================

st.sidebar.markdown('<div class="eyebrow">EnergySavvy AI</div>', unsafe_allow_html=True)
st.sidebar.markdown("### System Overview")

st.sidebar.markdown(
    """
    **Live data pipeline**

    1. IP-based location (ipapi.co)
    2. Open-Meteo live weather
    3. Real-time household simulator
    4. Forecast model (trained on UCI + Cairo data)
    5. Anomaly detection
    6. Recommendation engine
    """
)

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
st.sidebar.caption("EnergySavvy AI  |  Intelligent Systems")


# ============================================================
# PULL ONE LIVE TICK THROUGH THE FULL PIPELINE
# ============================================================

try:
    live_data = realtime_engine.get_live_data()
except Exception as e:
    st.error("Unable to retrieve live weather or location data.")
    st.exception(e)
    st.stop()

weather = live_data["weather"]
energy = live_data["energy"]
location = live_data["location"]

city = location.get("city", "Unknown")
latitude = location.get("lat")
longitude = location.get("lon")

temperature = weather.get("temperature_c")
humidity = weather.get("humidity_percent")
wind_speed = weather.get("wind_speed_kmh")

power_kw = energy.get("power_kw", 0.0)
voltage_v = energy.get("voltage_v", 0.0)
current_a = energy.get("current_a", 0.0)
appliances = energy.get("appliances", {})
reading_time = energy.get("timestamp")

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
# ROLLING CLIENT-SIDE CHART HISTORY (per browser session)
# ============================================================

if "chart_history" not in st.session_state:
    st.session_state.chart_history = []

st.session_state.chart_history.append(
    {
        "time": datetime.fromisoformat(reading_time),
        "power_kw": power_kw,
        "forecast_kw": forecast.get("prediction_kw"),
        "is_anomaly": anomaly.get("is_anomaly", False),
    }
)
st.session_state.chart_history = st.session_state.chart_history[-CHART_HISTORY_POINTS:]

chart_df = pd.DataFrame(st.session_state.chart_history)


# ============================================================
# LIVE STATUS ROW (KPI cards)
# ============================================================

st.markdown('<div class="section-title">Live System Status</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-caption">Updated automatically from live location, weather, and household data</div>',
    unsafe_allow_html=True,
)

forecast_value = f"{forecast['prediction_kw']:.3f} kW" if forecast.get("available") else "Initializing"
humidity_sub = f"{humidity:.0f}% relative humidity" if humidity is not None else "Humidity unavailable"
temperature_display = f"{float(temperature):.1f}\u00b0C" if temperature is not None else "N/A"

st.markdown(
    f"""
    <div class="kpi-row">
        <div class="kpi-card">
            <div class="kpi-label">Location</div>
            <div class="kpi-value">{city}</div>
            <div class="kpi-sub">Resolved from live IP geolocation</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Temperature</div>
            <div class="kpi-value">{temperature_display}</div>
            <div class="kpi-sub">{humidity_sub}</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Current Power Draw</div>
            <div class="kpi-value">{power_kw:.3f} kW</div>
            <div class="kpi-sub">{voltage_v:.1f} V &nbsp;|&nbsp; {current_a:.2f} A</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Forecast, Next Hour</div>
            <div class="kpi-value">{forecast_value}</div>
            <div class="kpi-sub">Random forest regression model</div>
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
        line=dict(color="#22b8cf", width=3),
        marker=dict(
            size=8,
            color=["#ef4444" if a else "#22b8cf" for a in chart_df["is_anomaly"]],
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
            line=dict(color="#94a3b8", width=2, dash="dash"),
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
    '<div class="section-caption">Live status of each simulated household appliance</div>',
    unsafe_allow_html=True,
)

if appliances:
    appliance_rows = [
        {
            "Appliance": name.replace("_", " ").title(),
            "Status": "Active" if data.get("on") else "Standby",
            "Power (kW)": data.get("power_kw", 0.0),
        }
        for name, data in appliances.items()
    ]
    appliance_df = pd.DataFrame(appliance_rows).sort_values("Power (kW)", ascending=False)

    a1, a2 = st.columns([1, 1])

    with a1:
        st.dataframe(appliance_df, use_container_width=True, hide_index=True)

    with a2:
        active_df = appliance_df[appliance_df["Power (kW)"] > 0]
        if not active_df.empty:
            bar_fig = go.Figure(
                go.Bar(
                    x=active_df["Power (kW)"],
                    y=active_df["Appliance"],
                    orientation="h",
                    marker_color="#22b8cf",
                )
            )
            bar_fig.update_layout(
                height=260,
                margin=dict(l=10, r=10, t=10, b=10),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#cbd5e1"),
                xaxis=dict(title="kW", gridcolor="rgba(148,163,184,0.15)"),
            )
            st.plotly_chart(bar_fig, use_container_width=True)
        else:
            st.info("No appliances are currently drawing power.")


# ============================================================
# ANOMALY DETECTION
# ============================================================

st.markdown('<div class="section-title">Anomaly Detection</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-caption">Statistical deviation from the trailing 24-hour baseline</div>',
    unsafe_allow_html=True,
)

if not anomaly.get("available"):
    st.info(anomaly.get("message"))
elif anomaly.get("is_anomaly"):
    st.markdown(
        f"""
        <div class="status-banner negative">
            <strong>Unusual consumption detected.</strong>
            Current draw is {power_kw:.3f} kW against a baseline of
            {anomaly['baseline_kw']:.3f} kW (z-score {anomaly['score']}).
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        f"""
        <div class="status-banner positive">
            <strong>Consumption is within the normal range.</strong>
            Current draw is {power_kw:.3f} kW against a baseline of
            {anomaly['baseline_kw']:.3f} kW (z-score {anomaly['score']}).
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# RECOMMENDATIONS
# ============================================================

st.markdown('<div class="section-title">Energy-Saving Recommendations</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-caption">Generated from current appliance activity and detected anomalies</div>',
    unsafe_allow_html=True,
)

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
            "Metric": "Anomaly Status",
            "Value": "Anomaly Detected" if anomaly.get("is_anomaly") else "Normal",
        },
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
    "above is generated live: real IP-based location, real Open-Meteo weather, and "
    "a live household simulator feeding the trained model."
)
st.caption(f"Last update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
