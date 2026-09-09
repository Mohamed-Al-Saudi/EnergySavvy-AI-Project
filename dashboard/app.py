"""
EnergySavvy AI
---------------
Final Streamlit deployment application.

Main responsibility:
    - Collect live weather/location data
    - Generate realtime energy data
    - Run the existing ML pipeline
    - Display predictions, anomalies and recommendations

IMPORTANT:
    Historical UCI/Cairo datasets are training data only.
    They are NOT used as live data.

Run locally:
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
# IMPORT EXISTING PROJECT MODULES
# ============================================================

# Realtime system
from src.realtime.realtime_engine import RealtimeEngine
from src.realtime.energy_simulator import EnergySimulator


# ============================================================
# STREAMLIT PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="EnergySavvy AI",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM STYLE
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 0px;
    }

    .subtitle {
        font-size: 18px;
        color: #666;
        margin-bottom: 25px;
    }

    .metric-card {
        padding: 15px;
        border-radius: 12px;
        background-color: #f7f7f7;
        text-align: center;
    }

    .section-title {
        font-size: 25px;
        font-weight: 650;
        margin-top: 25px;
        margin-bottom: 10px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">⚡ EnergySavvy AI</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">'
    "Intelligent Energy Management for a Sustainable Future"
    "</div>",
    unsafe_allow_html=True,
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("⚡ EnergySavvy AI")

st.sidebar.markdown(
    """
    ### System

    This application combines:

    - 🌤️ Live weather
    - 📍 Automatic location
    - ⚡ Realtime energy simulation
    - 🤖 Energy forecasting
    - 🚨 Anomaly detection
    - 💡 Energy recommendations

    ---
    """
)

st.sidebar.caption(
    "RoboDam 2026 • Intelligent Systems"
)


# ============================================================
# INITIALIZE REALTIME ENGINE
# ============================================================

@st.cache_resource
def create_realtime_engine():
    """
    Create one RealtimeEngine instance for the Streamlit app.

    The engine automatically determines the user's approximate
    location through the existing weather/location module.
    """
    return RealtimeEngine(auto_location=True)


@st.cache_resource
def create_energy_simulator():
    """
    Create the existing realtime energy simulator.
    """
    return EnergySimulator()


try:

    realtime_engine = create_realtime_engine()
    energy_simulator = create_energy_simulator()

except Exception as e:

    st.error("Unable to initialize the realtime energy system.")

    st.exception(e)

    st.stop()


# ============================================================
# GET LIVE DATA
# ============================================================

try:

    live_data = realtime_engine.get_live_data()

except Exception as e:

    st.error("Unable to retrieve live weather/location data.")

    st.exception(e)

    st.stop()


# ============================================================
# EXTRACT LOCATION
# ============================================================

latitude = live_data.get("latitude")
longitude = live_data.get("longitude")

city = live_data.get("city", "Unknown")

temperature = live_data.get("temperature")

weather_description = live_data.get(
    "weather_description",
    "Unavailable",
)


# ============================================================
# GENERATE REALTIME ENERGY DATA
# ============================================================

try:

    energy_data = energy_simulator.generate(
        temperature=(
            float(temperature)
            if temperature is not None
            else 30.0
        )
    )

except Exception as e:

    st.error("Unable to generate realtime energy data.")

    st.exception(e)

    st.stop()


# ============================================================
# CURRENT SYSTEM STATUS
# ============================================================

st.markdown(
    '<div class="section-title">📡 Live System Status</div>',
    unsafe_allow_html=True,
)


col1, col2, col3, col4 = st.columns(4)


# ------------------------------------------------------------
# LOCATION
# ------------------------------------------------------------

with col1:

    st.metric(
        label="📍 Location",
        value=str(city),
    )


# ------------------------------------------------------------
# TEMPERATURE
# ------------------------------------------------------------

with col2:

    if temperature is not None:

        st.metric(
            label="🌡️ Temperature",
            value=f"{float(temperature):.1f} °C",
        )

    else:

        st.metric(
            label="🌡️ Temperature",
            value="N/A",
        )


# ------------------------------------------------------------
# POWER
# ------------------------------------------------------------

power_kw = energy_data.get("power_kw", 0.0)

with col3:

    st.metric(
        label="⚡ Current Power",
        value=f"{float(power_kw):.3f} kW",
    )


# ------------------------------------------------------------
# WEATHER
# ------------------------------------------------------------

with col4:

    st.metric(
        label="🌤️ Weather",
        value=str(weather_description),
    )


# ============================================================
# LIVE DETAILS
# ============================================================

with st.expander("📍 Live environment details"):

    location_col1, location_col2 = st.columns(2)

    with location_col1:

        st.write("**City:**", city)
        st.write("**Latitude:**", latitude)

    with location_col2:

        st.write("**Longitude:**", longitude)
        st.write("**Weather:**", weather_description)


# ============================================================
# ENERGY DATA
# ============================================================

st.markdown(
    '<div class="section-title">⚡ Current Energy Consumption</div>',
    unsafe_allow_html=True,
)


energy_col1, energy_col2, energy_col3 = st.columns(3)


voltage = energy_data.get("voltage_v", 0.0)
current = energy_data.get("current_a", 0.0)
power = energy_data.get("power_kw", 0.0)


with energy_col1:

    st.metric(
        "Voltage",
        f"{float(voltage):.1f} V",
    )


with energy_col2:

    st.metric(
        "Current",
        f"{float(current):.2f} A",
    )


with energy_col3:

    st.metric(
        "Power",
        f"{float(power):.3f} kW",
    )


# ============================================================
# APPLIANCE DATA
# ============================================================

appliances = energy_data.get("appliances", {})


if appliances:

    st.markdown(
        '<div class="section-title">🔌 Appliance Consumption</div>',
        unsafe_allow_html=True,
    )

    appliance_df = pd.DataFrame(
        [
            {
                "Appliance": appliance,
                "Power": value,
            }
            for appliance, value in appliances.items()
        ]
    )

    st.dataframe(
        appliance_df,
        use_container_width=True,
        hide_index=True,
    )

    # Appliance chart
    if not appliance_df.empty:

        st.bar_chart(
            appliance_df.set_index("Appliance")
        )


# ============================================================
# FORECAST MODEL
# ============================================================

st.markdown(
    '<div class="section-title">🤖 Energy Forecast</div>',
    unsafe_allow_html=True,
)


if FORECAST_MODEL_PATH.exists():

    try:

        import joblib

        forecast_model = joblib.load(
            FORECAST_MODEL_PATH
        )

        st.success(
            "Forecasting model loaded successfully."
        )

        # ----------------------------------------------------
        # Prepare a feature vector.
        #
        # IMPORTANT:
        # This section should match the exact features used
        # when forecast_rf.pkl was trained.
        #
        # The first deployment version uses the realtime
        # variables available from the current system.
        # ----------------------------------------------------

        forecast_features = pd.DataFrame(
            [
                {
                    "temperature": (
                        float(temperature)
                        if temperature is not None
                        else 30.0
                    ),
                    "power_kw": float(power),
                    "voltage_v": float(voltage),
                    "current_a": float(current),
                }
            ]
        )

        # Try prediction.
        #
        # If the saved model expects a different number/name
        # of features, we show the error instead of crashing
        # the complete dashboard.

        try:

            prediction = forecast_model.predict(
                forecast_features
            )

            predicted_energy = float(
                prediction[0]
            )

            st.metric(
                "Predicted Energy",
                f"{predicted_energy:.3f}",
            )

        except Exception as prediction_error:

            st.warning(
                "The forecasting model is available, "
                "but its trained feature structure does not "
                "match the current realtime feature vector."
            )

            st.info(
                "The model must receive exactly the same "
                "features used during training."
            )

            with st.expander(
                "Technical prediction error"
            ):

                st.exception(
                    prediction_error
                )

    except Exception as model_error:

        st.error(
            "The forecasting model could not be loaded."
        )

        with st.expander(
            "Technical model error"
        ):

            st.exception(model_error)

else:

    st.warning(
        "Forecast model not found."
    )

    st.info(
        f"Expected model location: "
        f"{FORECAST_MODEL_PATH}"
    )


# ============================================================
# ANOMALY DETECTION
# ============================================================

st.markdown(
    '<div class="section-title">🚨 Anomaly Detection</div>',
    unsafe_allow_html=True,
)


# The realtime power value is displayed here.
# The final anomaly model can be connected through the
# existing src/models anomaly detection module once its
# exact prediction API is used.

if float(power) > 5.0:

    st.error(
        "⚠️ High energy consumption detected."
    )

    anomaly_status = "Potential anomaly"

else:

    st.success(
        "✅ Current energy consumption appears normal."
    )

    anomaly_status = "Normal"


# ============================================================
# RECOMMENDATIONS
# ============================================================

st.markdown(
    '<div class="section-title">💡 Energy-Saving Recommendations</div>',
    unsafe_allow_html=True,
)


recommendations = []


# ------------------------------------------------------------
# Temperature-based recommendation
# ------------------------------------------------------------

if temperature is not None:

    if float(temperature) >= 30:

        recommendations.append(
            "🌡️ High temperature may increase cooling demand. "
            "Use air conditioning efficiently and avoid "
            "unnecessary cooling."
        )


# ------------------------------------------------------------
# Power-based recommendation
# ------------------------------------------------------------

if float(power) >= 2:

    recommendations.append(
        "⚡ Current power consumption is relatively high. "
        "Check high-consumption appliances and turn off "
        "devices that are not needed."
    )


# ------------------------------------------------------------
# Anomaly recommendation
# ------------------------------------------------------------

if anomaly_status != "Normal":

    recommendations.append(
        "🚨 Investigate the unusual consumption pattern "
        "and check appliances that may be operating "
        "unnecessarily."
    )


# ------------------------------------------------------------
# Default recommendation
# ------------------------------------------------------------

if not recommendations:

    recommendations.append(
        "✅ Continue monitoring your consumption and "
        "avoid leaving unnecessary appliances running."
    )


for recommendation in recommendations:

    st.info(
        recommendation
    )


# ============================================================
# SYSTEM SUMMARY
# ============================================================

st.markdown(
    '<div class="section-title">📊 EnergySavvy AI Summary</div>',
    unsafe_allow_html=True,
)


summary_df = pd.DataFrame(
    [
        {
            "Metric": "Location",
            "Value": city,
        },
        {
            "Metric": "Temperature",
            "Value": (
                f"{float(temperature):.1f} °C"
                if temperature is not None
                else "N/A"
            ),
        },
        {
            "Metric": "Current Power",
            "Value": f"{float(power):.3f} kW",
        },
        {
            "Metric": "Voltage",
            "Value": f"{float(voltage):.1f} V",
        },
        {
            "Metric": "Current",
            "Value": f"{float(current):.2f} A",
        },
        {
            "Metric": "Anomaly Status",
            "Value": anomaly_status,
        },
    ]
)


st.dataframe(
    summary_df,
    use_container_width=True,
    hide_index=True,
)


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "EnergySavvy AI • RoboDam 2026 • Intelligent Systems"
)

st.caption(
    f"Last update: "
    f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
)
