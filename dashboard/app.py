# ============================================================================
# EnergySavvy AI - DEPLOYMENT DASHBOARD
# Existing dashboard revised for live-only production display.
#
# TRAINING ONLY:
#   - UCI Individual Household Electric Power Consumption (historical)
#   - Cairo Weather dataset (historical)
#
# PRODUCTION / JURY DEMO:
#   - Live governorate selection / IP auto-detection
#   - Open-Meteo live weather
#   - Stateful simulated household telemetry (temporary IoT replacement)
#   - Forecast model trained offline, but predictions use LIVE SIM telemetry
#   - Live anomaly detection
#   - Live, conditional recommendations
#
# ARCHITECTURE NOTE:
#   Every functional layer below (weather, IP location, the household
#   simulator, feature engineering, forecasting, anomaly detection,
#   recommendations, and bill calculation) is IMPORTED from src/ rather
#   than reimplemented here. This file only wires those modules together
#   and renders the Streamlit UI.
# ============================================================================

import base64
import io
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import qrcode
import streamlit as st

# ----------------------------------------------------------------------------
# Make `src` importable regardless of the working directory Streamlit Cloud
# launches from.
# ----------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.utils.helpers import MODEL_PATH  # noqa: E402
from src.realtime.weather_api import (  # noqa: E402
    GOVERNORATES,
    get_ip_location,
    get_live_weather,
)
from src.realtime.energy_simulator import (  # noqa: E402
    DEFAULT_CONTINUOUS_DEVICES,
    DEFAULT_VARIABLE_DEVICES,
    DEVICE_CATALOG,
    HouseholdEnergySimulator,
)
from src.features.household_features import (  # noqa: E402
    create_live_hourly_history,
    forecast_live_next_hours,
)
from src.models.forecasting import load_model  # noqa: E402
from src.models.anomaly_detection import live_anomaly  # noqa: E402
from src.recommendations.recommendation_engine import (  # noqa: E402
    generate_live_recommendations,
)
from src.billing.tariff import (  # noqa: E402
    estimate_actual_monthly_bill,
    estimate_monthly_from_devices,
    forecast_cost,
)

st.set_page_config(
    page_title="EnergySavvy AI | VoltAI",
    page_icon=":material/bolt:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------------
# Formal project visual + stable styling.
# The visual is embedded so the dashboard does not depend on a missing image
# file. The same SVG is also used as the browser favicon.
# ----------------------------------------------------------------------------
PROJECT_SVG = """<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="500" viewBox="0 0 1200 500">
<defs>
  <linearGradient id="bg" x1="0" y1="0" x2="1" y2="1">
    <stop offset="0" stop-color="#24134d"/><stop offset=".52" stop-color="#0b1528"/><stop offset="1" stop-color="#063c49"/>
  </linearGradient>
  <linearGradient id="line" x1="0" y1="0" x2="1" y2="0">
    <stop offset="0" stop-color="#a78bfa"/><stop offset=".55" stop-color="#22d3ee"/><stop offset="1" stop-color="#fb923c"/>
  </linearGradient>
</defs>
<rect width="1200" height="500" rx="36" fill="url(#bg)"/>
<circle cx="1010" cy="90" r="150" fill="#22d3ee" opacity=".08"/>
<circle cx="180" cy="390" r="180" fill="#a855f7" opacity=".08"/>
<path d="M70 350 C180 280 250 370 350 300 S520 220 610 300 S760 360 850 250 S1030 170 1130 230" fill="none" stroke="url(#line)" stroke-width="8" opacity=".9"/>
<path d="M95 385 H270 M930 385 H1110" stroke="#94a3b8" stroke-width="2" opacity=".35"/>
<g transform="translate(560 95)">
  <path d="M80 0 L25 105 H72 L42 180 L150 65 H102 L135 0 Z" fill="#fb923c"/>
  <circle cx="80" cy="80" r="105" fill="none" stroke="#22d3ee" stroke-width="3" opacity=".45"/>
</g>
<text x="70" y="90" fill="#ffffff" font-family="Arial,sans-serif" font-size="48" font-weight="700">EnergySavvy AI</text>
<text x="72" y="132" fill="#cbd5e1" font-family="Arial,sans-serif" font-size="21">Intelligent household energy management</text>
<text x="72" y="430" fill="#94a3b8" font-family="Arial,sans-serif" font-size="17">LIVE WEATHER  •  SIMULATED IoT POWER  •  FORECAST  •  ANOMALY  •  RECOMMENDATIONS</text>
</svg>"""

SVG_B64 = base64.b64encode(PROJECT_SVG.encode()).decode()
FAVICON_DATA = "data:image/svg+xml;base64," + SVG_B64

st.markdown(
    f"""<link rel="icon" type="image/svg+xml" href="{FAVICON_DATA}">
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600;700&display=swap');

    .stApp {{
        background:
            linear-gradient(115deg, rgba(43,24,78,.98) 0%, rgba(9,16,31,.99) 46%, rgba(4,58,70,.98) 100%);
        background-attachment: fixed;
        color: #e5e7eb;
        font-family: 'Inter', sans-serif;
    }}
    [data-testid="stHeader"] {{ background: rgba(7,11,22,.18); }}
    [data-testid="stAppViewContainer"] {{ background: transparent; }}
    [data-testid="stSidebar"] {{
        background: linear-gradient(180deg, rgba(24,17,46,.98), rgba(5,27,36,.98));
        border-right: 1px solid rgba(167,139,250,.20);
    }}
    .block-container {{ padding-top: 1.5rem; padding-bottom: 2rem; max-width: 1500px; }}
    .hero {{
        border: 1px solid rgba(255,255,255,.10);
        border-radius: 26px; overflow: hidden; margin-bottom: 18px;
        background: rgba(8,13,27,.36);
        box-shadow: 0 18px 50px rgba(0,0,0,.20);
    }}
    .hero img {{ width:100%; display:block; }}
    .section {{
        font-size: 23px; font-weight: 800; color:#f1f5f9;
        margin: 30px 0 14px; padding-bottom: 10px;
        border-bottom: 1px solid rgba(255,255,255,.11);
    }}
    .kpi {{
        background: linear-gradient(145deg, rgba(255,255,255,.105), rgba(255,255,255,.035));
        border: 1px solid rgba(255,255,255,.11);
        border-radius: 20px; padding: 17px 18px; min-height: 116px;
        box-shadow: 0 12px 30px rgba(0,0,0,.14);
    }}
    .kpi-label {{
        color:#94a3b8; font: 700 11px 'JetBrains Mono', monospace;
        text-transform:uppercase; letter-spacing:.08em;
    }}
    .kpi-value {{ color:#fff; font-size:28px; font-weight:800; margin:5px 0 3px; }}
    .kpi-sub {{ color:#cbd5e1; font: 500 12px 'JetBrains Mono', monospace; line-height:1.55; }}
    .pill {{
        display:inline-block; padding:5px 10px; border-radius:999px;
        font:700 10px 'JetBrains Mono', monospace; margin:3px 4px 3px 0;
    }}
    .pill-real {{ color:#86efac; border:1px solid rgba(34,197,94,.45); background:rgba(34,197,94,.10); }}
    .pill-sim {{ color:#fdba74; border:1px solid rgba(251,146,60,.45); background:rgba(251,146,60,.10); }}
    .pill-ai {{ color:#d8b4fe; border:1px solid rgba(168,85,247,.45); background:rgba(168,85,247,.10); }}
    .device-card {{
        background: rgba(255,255,255,.055); border:1px solid rgba(255,255,255,.09);
        border-radius:16px; padding:12px; margin-bottom:10px; min-height:92px;
    }}
    .device-title {{ font:700 12px 'JetBrains Mono',monospace; color:#cbd5e1; }}
    .device-power {{ font-size:20px; font-weight:800; margin-top:4px; }}
    .device-status {{ font:600 11px 'JetBrains Mono',monospace; color:#94a3b8; }}
    .rec {{
        background: linear-gradient(145deg, rgba(255,255,255,.09), rgba(255,255,255,.035));
        border:1px solid rgba(255,255,255,.10); border-radius:16px;
        padding:14px 16px; margin-bottom:10px;
    }}
    .rec-title {{ font:800 11px 'JetBrains Mono',monospace; text-transform:uppercase; }}
    .rec-msg {{ color:#e2e8f0; margin:7px 0; line-height:1.5; }}
    .rec-save {{ font:700 11px 'JetBrains Mono',monospace; color:#86efac; }}
    .small-note {{ color:#94a3b8; font:500 11px 'JetBrains Mono',monospace; line-height:1.6; }}
    .status-live {{ color:#86efac; }}
    .status-sim {{ color:#fdba74; }}
    </style>""",
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# Basic project information
# ----------------------------------------------------------------------------
TEAM_NAME = "VoltAI"
SUPERVISOR = "AbdelRahman Salem"
LEADER = "Mohamed Al-Saudi"
EMAIL = "mohamed.alsuadi2007@gmail.com"
LINKEDIN = "https://www.linkedin.com/in/mohamed-al-saudi-638274363/"
GITHUB = "https://github.com/Mohamed-Al-Saudi"
PROJECT_GITHUB = "https://github.com/Mohamed-Al-Saudi/EnergySavvy-AI-Project"
ORIGIN = "Cairo, Egypt"
PROJECT_NAME = "EnergySavvy AI"
# Public web-app URL entered by the user is used for the QR code.
DEFAULT_APP_URL = os.getenv("ENERGYSAVVY_APP_URL", "")
DESCRIPTION = (
    "EnergySavvy AI is a software-based intelligent energy management system "
    "that analyzes household electricity consumption data to understand usage "
    "patterns, forecast future consumption, detect unusual behavior, and "
    "generate data-driven recommendations."
)

st.markdown(
    f'<div class="hero"><img src="data:image/svg+xml;base64,{SVG_B64}" alt="EnergySavvy AI"></div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<span class="pill pill-real">LIVE WEATHER</span>'
    '<span class="pill pill-sim">SIMULATED IoT POWER</span>'
    '<span class="pill pill-ai">TRAINED AI MODELS</span>'
    '<span class="pill pill-real">LIVE PRODUCTION DISPLAY</span>',
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# Weather: Open-Meteo is primary and uses the selected governorate coordinates.
# wttr.in is a fallback, but NEVER hard-coded to Cairo. Logic lives in
# src.realtime.weather_api; this is just a thin Streamlit cache wrapper.
# ----------------------------------------------------------------------------
WEATHER_CACHE_SECONDS = 45


@st.cache_data(ttl=WEATHER_CACHE_SECONDS, show_spinner=False)
def get_current_weather(lat, lon):
    return get_live_weather(lat, lon)


# ----------------------------------------------------------------------------
# Sidebar: location, household devices, team.
# ----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## EnergySavvy AI")
    location_mode = st.radio(
        "Live location",
        ["Auto-detect", "Choose governorate"],
        index=0,
    )
    if location_mode == "Choose governorate":
        selected_governorate = st.selectbox(
            "Egyptian governorate",
            list(GOVERNORATES.keys()),
            index=list(GOVERNORATES.keys()).index("Cairo"),
        )
        selected_lat, selected_lon = GOVERNORATES[selected_governorate]
        selected_source = "Governorate coordinates"
    else:
        if "ip_location" not in st.session_state:
            st.session_state.ip_location = get_ip_location()
        ip = st.session_state.ip_location
        selected_governorate = ip["city"]
        selected_lat, selected_lon = ip["latitude"], ip["longitude"]
        selected_source = ip["source"]

    st.markdown("---")
    st.markdown("### Household devices")
    st.caption(
        "Choose which electrical appliances exist in this household, and "
        "whether each one behaves as a 24-hour device or a time-dependent "
        "device. The simulator, forecast, and bill below only use the "
        "devices selected here."
    )
    catalog_names = list(DEVICE_CATALOG.keys())

    selected_continuous = st.multiselect(
        "24-hour household devices",
        options=catalog_names,
        default=DEFAULT_CONTINUOUS_DEVICES,
        key="continuous_devices_select",
        help="Devices that remain available/active as part of the household "
             "baseline; their actual energy use follows a realistic duty cycle.",
    )
    remaining_for_variable = [n for n in catalog_names if n not in selected_continuous]
    default_variable = [n for n in DEFAULT_VARIABLE_DEVICES if n in remaining_for_variable]
    selected_variable = st.multiselect(
        "Time-dependent devices",
        options=remaining_for_variable,
        default=default_variable,
        key="variable_devices_select",
        help="Devices that switch on and off according to time of day, "
             "temperature, and stochastic household behavior.",
    )

    if not selected_continuous and not selected_variable:
        st.error("Select at least one household device to run the live dashboard.")
        st.stop()

    st.markdown("---")
    st.markdown("### Live pipeline")
    st.markdown(
        '<div class="small-note">'
        '1. Live location<br>2. Open-Meteo weather<br>3. Simulated IoT telemetry<br>'
        '4. Forecast inference<br>5. Live anomaly detection<br>6. Conditional recommendations'
        '</div>',
        unsafe_allow_html=True,
    )
    st.markdown("---")
    st.markdown("### Team")
    st.markdown(
        f"**{TEAM_NAME}**  \n"
        f"Supervisor: {SUPERVISOR}  \n"
        f"Leader & only member: {LEADER}  \n"
        f"From: {ORIGIN}"
    )
    st.markdown(f"[Email](mailto:{EMAIL})")
    st.markdown(f"[LinkedIn]({LINKEDIN})")
    st.markdown(f"[GitHub]({GITHUB})")
    st.markdown(f"[Project GitHub]({PROJECT_GITHUB})")

    st.markdown("---")
    st.markdown("### Web app QR code")
    st.caption("Enter the public Streamlit URL, then press Enter to generate a QR code for phone access.")

    with st.form("qr_form", clear_on_submit=False):
        entered_url = st.text_input(
            "Public web-app URL",
            value=st.session_state.get("qr_url", DEFAULT_APP_URL),
            placeholder="https://your-app.streamlit.app",
            label_visibility="visible",
        )
        generate_qr = st.form_submit_button("Generate QR Code", type="primary", use_container_width=True)

    if generate_qr:
        url = entered_url.strip()
        if url and not url.startswith(("http://", "https://")):
            url = "https://" + url
        if url.startswith(("http://", "https://")):
            st.session_state.qr_url = url
        else:
            st.session_state.qr_url = ""
            st.error("Please enter a valid public web-app URL.")

    qr_url = st.session_state.get("qr_url", DEFAULT_APP_URL)
    if qr_url:
        qr = qrcode.QRCode(version=1, box_size=7, border=3)
        qr.add_data(qr_url)
        qr.make(fit=True)
        qr_img = qr.make_image()
        qr_buffer = io.BytesIO()
        qr_img.save(qr_buffer, format="PNG")
        st.image(qr_buffer.getvalue(), width=190)
        st.caption("Scan with a phone camera to open EnergySavvy AI.")
        st.markdown(
            f'<div class="small-note">QR target: {qr_url}</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="small-note">No public URL entered yet. The QR code will appear here after you submit the URL.</div>',
            unsafe_allow_html=True,
        )

# ----------------------------------------------------------------------------
# Build the device configs used by the simulator/bill from the catalog +
# the user's sidebar selection.
# ----------------------------------------------------------------------------
continuous_devices_cfg = {
    name: {"kw": DEVICE_CATALOG[name]["kw"], "duty": DEVICE_CATALOG[name]["duty"]}
    for name in selected_continuous
}
variable_devices_cfg = {
    name: {"kw": DEVICE_CATALOG[name]["kw"], "profile": DEVICE_CATALOG[name]["profile"]}
    for name in selected_variable
}
TOTAL_DEVICES = len(continuous_devices_cfg) + len(variable_devices_cfg)

# ----------------------------------------------------------------------------
# Persistent live state.
# ----------------------------------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = []
if "hourly_live_history" not in st.session_state:
    st.session_state.hourly_live_history = None
if "last_hour_bucket" not in st.session_state:
    st.session_state.last_hour_bucket = None

# Recreate the simulator whenever the selected device list changes.
device_key = (tuple(sorted(selected_continuous)), tuple(sorted(selected_variable)))
if st.session_state.get("device_key") != device_key or "simulator" not in st.session_state:
    st.session_state.simulator = HouseholdEnergySimulator(continuous_devices_cfg, variable_devices_cfg)
    st.session_state.device_key = device_key
    st.session_state.hourly_live_history = None
    st.session_state.history = []

# Recreate the hourly warm-start only when location changes.
location_key = (round(float(selected_lat), 4), round(float(selected_lon), 4), selected_governorate)
if st.session_state.get("location_key") != location_key:
    st.session_state.location_key = location_key
    st.session_state.hourly_live_history = None
    st.session_state.history = []

# ----------------------------------------------------------------------------
# Render function. Streamlit fragment keeps generating live data continuously
# without requiring the jury to press Refresh.
# ----------------------------------------------------------------------------
def render_live_dashboard():
    now = datetime.now()
    weather = get_current_weather(selected_lat, selected_lon)
    energy = st.session_state.simulator.generate(temperature=weather["temperature_c"])
    live = {
        "timestamp": now.isoformat(),
        "time": now,
        "weather": weather,
        "energy": energy,
        "location": {
            "name": selected_governorate,
            "lat": selected_lat,
            "lon": selected_lon,
        },
    }

    st.session_state.history.append(live)
    st.session_state.history = st.session_state.history[-180:]

    # Warm-start 168 hourly points from the LIVE simulator, not old datasets.
    if st.session_state.hourly_live_history is None:
        st.session_state.hourly_live_history = create_live_hourly_history(
            st.session_state.simulator, weather["temperature_c"]
        )
        st.session_state.last_hour_bucket = now.replace(minute=0, second=0, microsecond=0)

    # Only add a new hourly point when a real clock hour changes.
    current_bucket = now.replace(minute=0, second=0, microsecond=0)
    if current_bucket > st.session_state.last_hour_bucket:
        d_hour = st.session_state.simulator.generate(temperature=weather["temperature_c"], when=current_bucket)
        st.session_state.hourly_live_history.loc[current_bucket] = d_hour["power_kw"]
        st.session_state.hourly_live_history = st.session_state.hourly_live_history.tail(168)
        st.session_state.last_hour_bucket = current_bucket

    # ------------------------------------------------------------------------
    # LIVE SYSTEM STATUS
    # ------------------------------------------------------------------------
    e = live["energy"]
    w = live["weather"]
    active_variable = sum(
        1 for name in variable_devices_cfg if e["appliances"].get(name, {}).get("on")
    )
    active_total = len(continuous_devices_cfg) + active_variable

    st.markdown('<div class="section">Live System Dashboard</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            f'<div class="kpi"><div class="kpi-label">Location</div>'
            f'<div class="kpi-value">{selected_governorate}</div>'
            f'<div class="kpi-sub">{selected_lat:.4f}, {selected_lon:.4f}<br>{selected_source}</div></div>',
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f'<div class="kpi"><div class="kpi-label">Live weather</div>'
            f'<div class="kpi-value">{w["temperature_c"]:.1f}°C</div>'
            f'<div class="kpi-sub">{w["weather_desc"]} | Feels {w["feels_like_c"]:.1f}°C<br>'
            f'Humidity {w["humidity_percent"]}% | Wind {w["wind_speed_kmh"]:.1f} km/h<br>'
            f'<span class="status-live">{w["source"]}</span></div></div>',
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f'<div class="kpi"><div class="kpi-label">Instantaneous power</div>'
            f'<div class="kpi-value">{e["power_kw"]:.2f} kW</div>'
            f'<div class="kpi-sub">{e["power_w"]:.0f} W | {e["voltage_v"]:.1f} V | {e["current_a"]:.2f} A<br>'
            f'<span class="status-sim">SIM IoT telemetry • {active_total}/{TOTAL_DEVICES} active</span></div></div>',
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            f'<div class="kpi"><div class="kpi-label">Live stream</div>'
            f'<div class="kpi-value">{len(st.session_state.history)} points</div>'
            f'<div class="kpi-sub">Updated {now.strftime("%H:%M:%S")}<br>'
            f'<span class="status-live">Auto-refresh every 5 seconds</span></div></div>',
            unsafe_allow_html=True,
        )

    # ------------------------------------------------------------------------
    # LIVE CONSUMPTION
    # ------------------------------------------------------------------------
    st.markdown('<div class="section">Live Consumption Dashboard</div>', unsafe_allow_html=True)
    plot_df = pd.DataFrame([
        {"time": x["time"], "power_kw": x["energy"]["power_kw"]}
        for x in st.session_state.history
    ])
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=plot_df["time"], y=plot_df["power_kw"], mode="lines+markers",
        name="Live simulated power", line=dict(color="#fb923c", width=3),
        marker=dict(size=4), fill="tozeroy", fillcolor="rgba(251,146,60,.12)"
    ))
    fig.update_layout(
        height=360, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#cbd5e1"), margin=dict(l=10,r=10,t=20,b=10),
        hovermode="x unified", xaxis_title=None, yaxis_title="Power (kW)",
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True, key=f"live_power_{len(st.session_state.history)}")
    st.caption("Production display: every point above is generated from the live simulator during this session. No UCI consumption rows are displayed.")

    # ------------------------------------------------------------------------
    # HOUSEHOLD DEVICE DASHBOARD
    # ------------------------------------------------------------------------
    st.markdown(f'<div class="section">Household Device Dashboard — {TOTAL_DEVICES} Devices</div>', unsafe_allow_html=True)
    left, right = st.columns(2)
    with left:
        st.markdown("#### 24-hour household devices")
        st.caption("These devices remain available/active as part of the household baseline; their actual energy contribution uses realistic duty cycles.")
        if not continuous_devices_cfg:
            st.markdown('<div class="small-note">No 24-hour devices selected in the sidebar.</div>', unsafe_allow_html=True)
        for name in continuous_devices_cfg:
            d = e["appliances"][name]
            st.markdown(
                f'<div class="device-card"><div class="device-title">{name.upper()}</div>'
                f'<div class="device-power" style="color:#86efac;">{d["power_kw"]:.3f} kW</div>'
                f'<div class="device-status">ACTIVE • 24-HOUR GROUP • instantaneous cycle {"ON" if d["instant_active"] else "LOW"}'
                f'</div></div>', unsafe_allow_html=True
            )
    with right:
        st.markdown("#### Time-dependent devices")
        st.caption("These devices switch according to time of day, temperature, and stochastic household behavior.")
        if not variable_devices_cfg:
            st.markdown('<div class="small-note">No time-dependent devices selected in the sidebar.</div>', unsafe_allow_html=True)
        for name in variable_devices_cfg:
            d = e["appliances"][name]
            state_color = "#86efac" if d["on"] else "#64748b"
            st.markdown(
                f'<div class="device-card"><div class="device-title">{name.upper()}</div>'
                f'<div class="device-power" style="color:{state_color};">{d["power_kw"]:.3f} kW</div>'
                f'<div class="device-status">{"ON" if d["on"] else "OFF"} • probability {d["prob"]:.2f} • TIME-DEPENDENT'
                f'</div></div>', unsafe_allow_html=True
            )

    # ------------------------------------------------------------------------
    # 1. FORECAST + BILL
    # ------------------------------------------------------------------------
    st.markdown('<div class="section">1. Forecast Future Consumption & Monthly Bill</div>', unsafe_allow_html=True)
    model = load_model()
    hourly = st.session_state.hourly_live_history
    preds_24h = forecast_live_next_hours(model, hourly, hours=24)
    preds = preds_24h[:6]
    forecast_times = [hourly.index[-1] + timedelta(hours=i) for i in range(1, 7)]

    cf1, cf2 = st.columns([2.1, 1])
    with cf1:
        figf = go.Figure()
        figf.add_trace(go.Scatter(
            x=hourly.index[-24:], y=hourly.values[-24:],
            mode="lines", name="Live recent baseline",
            line=dict(color="#22d3ee", width=2)
        ))
        figf.add_trace(go.Scatter(
            x=forecast_times, y=preds,
            mode="lines+markers", name="Next 6h forecast",
            line=dict(color="#a78bfa", width=3)
        ))
        figf.update_layout(
            height=340, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#cbd5e1"), margin=dict(l=10,r=10,t=20,b=10),
            hovermode="x unified", yaxis_title="Power (kW)"
        )
        st.plotly_chart(figf, use_container_width=True, key=f"forecast_{len(st.session_state.history)}")
        st.caption("Forecast input = live simulated household telemetry. The historical training dataset is not plotted or used as production observations.")

    daily_kwh, monthly_kwh, monthly_bill, bill_rows = estimate_monthly_from_devices(
        st.session_state.simulator, w["temperature_c"]
    )
    with cf2:
        st.metric("Next-hour forecast", f"{preds[0]:.2f} kW", f"{preds[0]-e['power_kw']:+.2f} kW vs now")
        st.metric("Expected daily use", f"{daily_kwh:.1f} kWh")
        st.metric("Projected monthly use", f"{monthly_kwh:.0f} kWh")
        st.metric("Projected monthly bill", f"{monthly_bill:,.0f} EGP")
        st.caption("Estimate = device power × expected daily operating hours × 30 days, then residential tier pricing.")

    # -- New: actual bill + next-hour / next-24h bill, from the forecast model --
    actual_monthly_kwh, actual_monthly_bill = estimate_actual_monthly_bill(e["power_kw"])
    next_hour_kwh, next_hour_cost, eff_rate = forecast_cost([preds_24h[0]], monthly_kwh)
    next_24h_kwh, next_24h_cost, _ = forecast_cost(preds_24h, monthly_kwh)

    bill1, bill2, bill3 = st.columns(3)
    with bill1:
        st.metric(
            "Actual bill (current live rate)",
            f"{actual_monthly_bill:,.0f} EGP",
            help="What the household would pay if the current instantaneous "
                 "live demand held constant for 24h/day across 30 days.",
        )
    with bill2:
        st.metric(
            "Next-hour bill",
            f"{next_hour_cost:,.2f} EGP",
            f"{next_hour_kwh:.2f} kWh forecast",
        )
    with bill3:
        st.metric(
            "Next-24h bill",
            f"{next_24h_cost:,.2f} EGP",
            f"{next_24h_kwh:.1f} kWh forecast",
        )
    st.caption(
        f"Next-hour and next-24h bills apply the household's current effective "
        f"tariff rate ({eff_rate:.2f} EGP/kWh, from the Egyptian residential "
        f"tiers below) to the forecast model's live predictions."
    )

    with st.expander("Monthly bill calculation — live household assumptions"):
        bill_df = pd.DataFrame(bill_rows, columns=["Device", "Rated kW", "Expected hours/day", "Expected kWh/day"])
        st.dataframe(bill_df.round(3), use_container_width=True, hide_index=True)
        st.markdown(
            "**Tariff assumptions used in this demo:** "
            "up to 350 kWh = 1.72 EGP/kWh; 351–650 = 2.18; "
            "651–1000 = 2.40; above 1000 = 2.74 EGP/kWh on the full consumption."
        )

    # ------------------------------------------------------------------------
    # 2. ANOMALY
    # ------------------------------------------------------------------------
    st.markdown('<div class="section">2. Detect Unusual Behavior</div>', unsafe_allow_html=True)
    power_history = [x["energy"]["power_kw"] for x in st.session_state.history]
    is_anomaly, z, baseline = live_anomaly(e["power_kw"], power_history)
    ca, cb = st.columns([1, 2])
    with ca:
        if is_anomaly:
            st.error(f"LIVE ANOMALY DETECTED\n\n{e['power_kw']:.2f} kW  |  z = {z:.2f}")
        else:
            st.success(f"LIVE NORMAL\n\n{e['power_kw']:.2f} kW  |  z = {z:.2f}")
        st.metric("Recent live baseline", f"{baseline:.2f} kW")
        st.metric("Deviation", f"{e['power_kw'] - baseline:+.2f} kW")
    with cb:
        st.markdown(
            f'<div class="kpi"><div class="kpi-label">Live anomaly explanation</div>'
            f'<div class="kpi-value" style="font-size:20px;">{"Attention required" if is_anomaly else "Within recent live range"}</div>'
            f'<div class="kpi-sub">The decision is based on the current session’s rolling live telemetry. '
            f'It is recalculated on every update, so the status can change between normal and unusual behavior.</div></div>',
            unsafe_allow_html=True,
        )

    # ------------------------------------------------------------------------
    # 3. RECOMMENDATIONS — messages only, no old recommendation table
    # ------------------------------------------------------------------------
    st.markdown('<div class="section">3. Data-Driven Recommendations</div>', unsafe_allow_html=True)
    recs = generate_live_recommendations(live, is_anomaly, z, monthly_kwh)
    if not recs:
        st.success("No action is required right now. Current live conditions are within the configured energy-saving thresholds.")
    else:
        for title, severity, message, saving in recs:
            if severity == "High":
                border = "#ef4444"
            elif severity == "Medium":
                border = "#f59e0b"
            else:
                border = "#22c55e"
            st.markdown(
                f'<div class="rec" style="border-left:4px solid {border};">'
                f'<div class="rec-title">{title} • {severity}</div>'
                f'<div class="rec-msg">{message}</div>'
                f'<div class="rec-save">Estimated saving: {saving:.2f} kWh per applicable day/event</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    # ------------------------------------------------------------------------
    # Snapshot / production transparency
    # ------------------------------------------------------------------------
    st.markdown('<div class="section">System Snapshot</div>', unsafe_allow_html=True)
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Governorate", selected_governorate)
    s2.metric("Temperature", f'{w["temperature_c"]:.1f}°C')
    s3.metric("Live power", f'{e["power_kw"]:.2f} kW')
    s4.metric("Monthly bill projection", f'{monthly_bill:,.0f} EGP')

    st.markdown(
        '<div class="small-note">'
        '<b>Production data boundary:</b> UCI Household and historical Cairo Weather datasets were used for offline model training/validation only. '
        'The jury-facing dashboard displays current location, live weather, and newly generated simulated household telemetry. '
        'The simulator is a temporary replacement for real IoT sensors and can later be connected to physical meters.'
        '</div>',
        unsafe_allow_html=True,
    )

# ----------------------------------------------------------------------------
# Continuous refresh. Streamlit fragment avoids the old "wait 1 minute" issue.
# Fallback for older Streamlit versions: show a manual refresh button.
# ----------------------------------------------------------------------------
if hasattr(st, "fragment"):
    @st.fragment(run_every="5s")
    def live_fragment():
        render_live_dashboard()
    live_fragment()
else:
    render_live_dashboard()
    if st.button("Refresh live data", type="primary"):
        st.rerun()

# ----------------------------------------------------------------------------
# Footer
# ----------------------------------------------------------------------------
st.markdown("---")
st.markdown(
    f'<div class="small-note">'
    f'<b>{PROJECT_NAME}</b> • Team <b>{TEAM_NAME}</b> • Supervisor {SUPERVISOR} • '
    f'Leader & only member {LEADER} • {ORIGIN}<br>'
    f'{DESCRIPTION}<br><br>'
    f'<a href="{LINKEDIN}" target="_blank">LinkedIn</a> &nbsp;|&nbsp; '
    f'<a href="{GITHUB}" target="_blank">GitHub</a> &nbsp;|&nbsp; '
    f'<a href="{PROJECT_GITHUB}" target="_blank">Project GitHub</a> &nbsp;|&nbsp; '
    f'{EMAIL}'
    f'</div>',
    unsafe_allow_html=True,
)
