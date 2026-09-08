"""
EnergySavvy AI

Final Streamlit Deployment Application.

Responsibilities:
    - Automatic location detection via IP geolocation
    - Live weather data acquisition from Open-Meteo
    - Realtime household energy simulation
    - Energy forecasting using trained models
    - Anomaly detection and intelligent recommendations

Note:
    Historical UCI and Cairo datasets are used exclusively for model training.
    Live operation relies entirely on realtime data sources.

Execution:
    streamlit run app.py
"""

from pathlib import Path
from datetime import datetime

import pandas as pd
import streamlit as st

# ============================================================
# PROJECT PATHS
# ============================================================

ROOT_DIR = Path(__file__).resolve().parent
MODEL_DIR = ROOT_DIR / "models"
FORECAST_MODEL_PATH = MODEL_DIR / "forecast_rf.pkl"

# ============================================================
# CORE MODULES
# ============================================================

from src.realtime.realtime_engine import RealtimeEngine
from src.realtime.energy_simulator import EnergySimulator

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="EnergySavvy AI | Intelligent Energy Management",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# STYLING
# ============================================================

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

   .main-title {
        font-size: 38px;
        font-weight: 700;
        letter-spacing: -0.02em;
        color: #0F172A;
        margin-bottom: 4px;
    }

   .subtitle {
        font-size: 17px;
        font-weight: 400;
        color: #64748B;
        margin-bottom: 28px;
        letter-spacing: 0.01em;
    }

   .section-title {
        font-size: 20px;
        font-weight: 600;
        color: #0F172A;
        margin-top: 32px;
        margin-bottom: 16px;
        padding-bottom: 8px;
        border-bottom: 1px solid #E2E8F0;
    }

   .status-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 16px;
    }

    div[data-testid="metric-container"] {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        padding: 16px 18px;
        border-radius: 12px;
    }

    div[data-testid="metric-container"] > label {
        color: #64748B!important;
        font-size: 13px!important;
        font-weight: 500!important;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }

    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# HEADER
# ============================================================

st.markdown('<div class="main-title">EnergySavvy AI</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Intelligent Energy Management for a Sustainable Future</div>',
    unsafe_allow_html=True,
)

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.title("EnergySavvy AI")
    st.markdown("**RoboDam 2026**")
    st.markdown("Intelligent Systems Track")
    st.divider()
    st.markdown(
        """
        **System Overview**

        This platform integrates multiple realtime data sources:

        - Automatic location detection
        - Live weather acquisition
        - Realtime energy simulation
        - Energy consumption forecasting
        - Anomaly detection
        - Intelligent recommendations

        Historical datasets are utilized solely for model training.
        Live operation is based entirely on realtime inputs.
        """
    )
    st.divider()
    st.caption("EnergySavvy AI | Version 1.0")
    st.caption(f"Date: {datetime.now().strftime('%Y-%m-%d')}")

# ============================================================
# INITIALIZE ENGINE
# ============================================================

@st.cache_resource(show_spinner=False)
def create_realtime_engine():
    return RealtimeEngine(auto_location=True)

@st.cache_resource(show_spinner=False)
def create_energy_simulator():
    return EnergySimulator()

try:
    realtime_engine = create_realtime_engine()
    energy_simulator = create_energy_simulator()
except Exception as e:
    st.error("System Initialization Failed")
    st.exception(e)
    st.stop()

# ============================================================
# RETRIEVE LIVE DATA
# ============================================================

try:
    with st.spinner("Acquiring live system data..."):
        live_data = realtime_engine.get_live_data()
except Exception as e:
    st.error("Unable to retrieve live data. Please check network connectivity.")
    st.exception(e)
    st.stop()

# Extract fields from unified realtime engine structure
weather_data = live_data.get("weather", {})
energy_data = live_data.get("energy", {})
location_data = live_data.get("location", {})
timestamp = live_data.get("timestamp")

city = location_data.get("city", "Cairo")
latitude = location_data.get("lat") or weather_data.get("latitude")
longitude = location_data.get("lon") or weather_data.get("longitude")

temperature_c = weather_data.get("temperature_c")
humidity_percent = weather_data.get("humidity_percent")
wind_speed = weather_data.get("wind_speed_kmh")

voltage_v = energy_data.get("voltage_v", 0.0)
current_a = energy_data.get("current_a", 0.0)
power_kw = energy_data.get("power_kw", 0.0)
appliances = energy_data.get("appliances", {})

# ============================================================
# LIVE SYSTEM STATUS
# ============================================================

st.markdown('<div class="section-title">Live System Status</div>', unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(label="Location", value=str(city))

with col2:
    st.metric(
        label="Temperature",
        value=f"{temperature_c:.1f} °C" if temperature_c is not None else "N/A",
        delta=f"Humidity {humidity_percent:.0f}%" if humidity_percent is not None else None,
        delta_color="off"
    )

with col3:
    st.metric(label="Current Power", value=f"{power_kw:.3f} kW")

with col4:
    st.metric(
        label="Wind Speed",
        value=f"{wind_speed:.1f} km/h" if wind_speed is not None else "N/A"
    )

with st.expander("View Detailed Environment Information"):
    dcol1, dcol2 = st.columns(2)
    with dcol1:
        st.write(f"**City:** {city}")
        st.write(f"**Latitude:** {latitude}")
        st.write(f"**Longitude:** {longitude}")
    with dcol2:
        st.write(f"**Temperature:** {temperature_c} °C" if temperature_c else "Temperature: N/A")
        st.write(f"**Humidity:** {humidity_percent} %" if humidity_percent else "Humidity: N/A")
        st.write(f"**Timestamp:** {timestamp}")

# ============================================================
# ENERGY CONSUMPTION
# ============================================================

st.markdown('<div class="section-title">Current Energy Consumption</div>', unsafe_allow_html=True)

e_col1, e_col2, e_col3 = st.columns(3)

with e_col1:
    st.metric("Voltage", f"{float(voltage_v):.1f} V")
with e_col2:
    st.metric("Current", f"{float(current_a):.2f} A")
with e_col3:
    st.metric("Power", f"{float(power_kw):.3f} kW")

# ============================================================
# APPLIANCE BREAKDOWN
# ============================================================

if appliances:
    st.markdown('<div class="section-title">Appliance-Level Consumption</div>', unsafe_allow_html=True)

    appliance_df = pd.DataFrame(
        [{"Appliance": k, "Power (kW)": float(v)} for k, v in appliances.items()]
    ).sort_values("Power (kW)", ascending=False)

    t_col1, t_col2 = st.columns([1.2, 1])

    with t_col1:
        st.dataframe(appliance_df, use_container_width=True, hide_index=True)

    with t_col2:
        if not appliance_df.empty:
            st.bar_chart(appliance_df.set_index("Appliance"))

# ============================================================
# FORECASTING
# ============================================================

st.markdown('<div class="section-title">Energy Forecasting</div>', unsafe_allow_html=True)

if FORECAST_MODEL_PATH.exists():
    try:
        import joblib
        forecast_model = joblib.load(FORECAST_MODEL_PATH)
        st.success("Forecasting model loaded successfully.")

        forecast_features = pd.DataFrame([{
            "temperature": float(temperature_c) if temperature_c is not None else 30.0,
            "power_kw": float(power_kw),
            "voltage_v": float(voltage_v),
            "current_a": float(current_a),
        }])

        try:
            prediction = forecast_model.predict(forecast_features)
            st.metric("Predicted Energy Consumption", f"{float(prediction[0]):.3f} kW")

        except Exception as pred_err:
            st.warning(
                "The forecasting model is available, but the current feature vector "
                "does not match the structure used during training. Please ensure "
                "feature alignment."
            )
            with st.expander("Technical Details"):
                st.exception(pred_err)

    except Exception as model_err:
        st.error("The forecasting model could not be loaded.")
        with st.expander("Technical Details"):
            st.exception(model_err)
else:
    st.warning("Forecast model not found.")
    st.info(f"Expected location: {FORECAST_MODEL_PATH}")

# ============================================================
# ANOMALY DETECTION
# ============================================================

st.markdown('<div class="section-title">Anomaly Detection</div>', unsafe_allow_html=True)

if float(power_kw) > 5.0:
    st.error("High energy consumption detected. Potential anomaly identified.")
    anomaly_status = "Anomaly Detected"
else:
    st.success("Current energy consumption is within normal operating range.")
    anomaly_status = "Normal"

# ============================================================
# RECOMMENDATIONS
# ============================================================

st.markdown('<div class="section-title">Energy Optimization Recommendations</div>', unsafe_allow_html=True)

recommendations = []

if temperature_c is not None and float(temperature_c) >= 30:
    recommendations.append(
        "Elevated ambient temperature detected. Cooling demand is expected to increase. "
        "It is recommended to optimize air conditioning usage and ensure efficient thermal insulation."
    )

if float(power_kw) >= 2.0:
    recommendations.append(
        "Current power consumption is above nominal levels. Review active appliances "
        "and deactivate non-essential devices to reduce load."
    )

if anomaly_status!= "Normal":
    recommendations.append(
        "An anomalous consumption pattern has been identified. A detailed inspection "
        "of appliance operation is advised."
    )

if not recommendations:
    recommendations.append(
        "System is operating efficiently. Continue monitoring and avoid unnecessary appliance usage."
    )

for rec in recommendations:
    st.info(rec)

# ============================================================
# SUMMARY
# ============================================================

st.markdown('<div class="section-title">System Summary</div>', unsafe_allow_html=True)

summary_df = pd.DataFrame([
    {"Metric": "Location", "Value": city},
    {"Metric": "Latitude / Longitude", "Value": f"{latitude}, {longitude}"},
    {"Metric": "Temperature", "Value": f"{temperature_c:.1f} °C" if temperature_c is not None else "N/A"},
    {"Metric": "Humidity", "Value": f"{humidity_percent:.0f} %" if humidity_percent is not None else "N/A"},
    {"Metric": "Wind Speed", "Value": f"{wind_speed:.1f} km/h" if wind_speed is not None else "N/A"},
    {"Metric": "Voltage", "Value": f"{voltage_v:.1f} V"},
    {"Metric": "Current", "Value": f"{current_a:.2f} A"},
    {"Metric": "Power", "Value": f"{power_kw:.3f} kW"},
    {"Metric": "Anomaly Status", "Value": anomaly_status},
])

st.dataframe(summary_df, use_container_width=True, hide_index=True)

# ============================================================
# FOOTER
# ============================================================

st.divider()
c1, c2 = st.columns([2, 1])
with c1:
    st.caption("EnergySavvy AI | RoboDam 2026 | Intelligent Systems")
with c2:
    st.caption(f"Last Update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
