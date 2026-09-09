from pathlib import Path
import sys
import time

import pandas as pd
import streamlit as st

# ---------------------------------------------------------
# Project path
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.realtime.realtime_engine import RealtimeEngine
from src.realtime.inference_engine import InferenceEngine


# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------

st.set_page_config(
    page_title="EnergySavvy AI",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------

st.markdown(
    """
    <style>

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1500px;
    }

    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }

    .subtitle {
        color: #6b7280;
        font-size: 1rem;
        margin-bottom: 2rem;
    }

    .section-title {
        font-size: 1.45rem;
        font-weight: 650;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }

    .status-card {
        padding: 1rem;
        border-radius: 12px;
        border: 1px solid #e5e7eb;
        background: #ffffff;
        margin-bottom: 0.75rem;
    }

    .recommendation-card {
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #374151;
        background: #f9fafb;
        margin-bottom: 0.75rem;
    }

    .appliance-on {
        padding: 0.65rem;
        border-radius: 8px;
        background: #f3f4f6;
        margin-bottom: 0.4rem;
    }

    .appliance-off {
        padding: 0.65rem;
        border-radius: 8px;
        background: #fafafa;
        color: #9ca3af;
        margin-bottom: 0.4rem;
    }

    .small-text {
        color: #6b7280;
        font-size: 0.85rem;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# Session state
# ---------------------------------------------------------

if "realtime_engine" not in st.session_state:
    st.session_state.realtime_engine = RealtimeEngine(
        auto_location=True
    )

if "inference_engine" not in st.session_state:
    st.session_state.inference_engine = InferenceEngine(
        model_path=str(
            PROJECT_ROOT / "models" / "forecast_rf.pkl"
        )
    )

if "power_history" not in st.session_state:
    st.session_state.power_history = []

if "timestamps" not in st.session_state:
    st.session_state.timestamps = []

if "auto_refresh" not in st.session_state:
    st.session_state.auto_refresh = True

if "refresh_seconds" not in st.session_state:
    st.session_state.refresh_seconds = 5


engine = st.session_state.realtime_engine
inference = st.session_state.inference_engine


# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------

with st.sidebar:

    st.markdown("## EnergySavvy AI")

    st.markdown(
        "Real-time intelligent household energy management."
    )

    st.divider()

    st.markdown("### Dashboard controls")

    st.session_state.auto_refresh = st.toggle(
        "Automatic refresh",
        value=st.session_state.auto_refresh,
    )

    st.session_state.refresh_seconds = st.slider(
        "Refresh interval (seconds)",
        min_value=2,
        max_value=30,
        value=st.session_state.refresh_seconds,
    )

    st.divider()

    st.markdown("### Location")

    manual_location = st.toggle(
        "Use manual location",
        value=False,
    )

    if manual_location:

        latitude = st.number_input(
            "Latitude",
            value=float(engine.latitude),
            format="%.4f",
        )

        longitude = st.number_input(
            "Longitude",
            value=float(engine.longitude),
            format="%.4f",
        )

        city = st.text_input(
            "City",
            value=engine.city,
        )

        if st.button("Update location", use_container_width=True):

            engine.update_location(
                latitude=latitude,
                longitude=longitude,
                city=city,
            )

            st.rerun()

    else:

        st.markdown(
            f"Current location: **{engine.city}**"
        )

        st.markdown(
            f"Latitude: `{engine.latitude:.4f}`"
        )

        st.markdown(
            f"Longitude: `{engine.longitude:.4f}`"
        )

    st.divider()

    st.markdown("### System status")

    st.success("Real-time engine active")

    st.info("Forecast model loaded")


# ---------------------------------------------------------
# Header
# ---------------------------------------------------------

st.markdown(
    '<div class="main-title">EnergySavvy AI</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">'
    "Intelligent real-time household energy management"
    "</div>",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# Get live data
# ---------------------------------------------------------

try:

    live_data = engine.get_live_data()

    ai_result = inference.process(
        live_data
    )

except Exception as error:

    st.error(
        f"Unable to retrieve real-time system data: {error}"
    )

    st.stop()


weather = live_data["weather"]
energy = live_data["energy"]
location = live_data["location"]

forecast = ai_result["forecast"]
anomaly = ai_result["anomaly"]
recommendations = ai_result["recommendations"]


# ---------------------------------------------------------
# Live status
# ---------------------------------------------------------

st.markdown(
    '<div class="section-title">Live system status</div>',
    unsafe_allow_html=True,
)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Current power",
        f"{energy['power_kw']:.2f} kW",
    )

with col2:
    st.metric(
        "Voltage",
        f"{energy['voltage_v']:.1f} V",
    )

with col3:
    st.metric(
        "Current",
        f"{energy['current_a']:.2f} A",
    )

with col4:
    st.metric(
        "Temperature",
        f"{weather['temperature_c']:.1f} °C",
    )


# ---------------------------------------------------------
# Weather
# ---------------------------------------------------------

st.markdown(
    '<div class="section-title">Live environmental conditions</div>',
    unsafe_allow_html=True,
)

weather_col1, weather_col2, weather_col3, weather_col4 = st.columns(4)

with weather_col1:
    st.metric(
        "Location",
        location["city"],
    )

with weather_col2:
    st.metric(
        "Temperature",
        f"{weather['temperature_c']:.1f} °C",
    )

with weather_col3:
    st.metric(
        "Humidity",
        f"{weather['humidity_percent']:.0f} %",
    )

with weather_col4:
    st.metric(
        "Wind speed",
        f"{weather.get('wind_speed_kmh', 0):.1f} km/h",
    )


# ---------------------------------------------------------
# Appliance states
# ---------------------------------------------------------

st.markdown(
    '<div class="section-title">Appliance states</div>',
    unsafe_allow_html=True,
)

appliances = energy["appliances"]

appliance_columns = st.columns(4)

for index, (name, data) in enumerate(
    appliances.items()
):

    column = appliance_columns[
        index % 4
    ]

    display_name = name.replace(
        "_",
        " ",
    ).title()

    state = "ON" if data["on"] else "OFF"

    css_class = (
        "appliance-on"
        if data["on"]
        else "appliance-off"
    )

    with column:

        st.markdown(
            f"""
            <div class="{css_class}">
                <strong>{display_name}</strong><br>
                {state}<br>
                <span class="small-text">
                    {data['power_kw']:.3f} kW
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------
# AI outputs
# ---------------------------------------------------------

st.markdown(
    '<div class="section-title">AI analysis</div>',
    unsafe_allow_html=True,
)

ai_col1, ai_col2 = st.columns(2)


# ---------------------------------------------------------
# Forecast
# ---------------------------------------------------------

with ai_col1:

    st.markdown("### Consumption forecast")

    if forecast["available"]:

        st.metric(
            "Predicted consumption",
            f"{forecast['prediction_kw']:.2f} kW",
        )

        st.success(
            "Forecast generated from the trained Random Forest model."
        )

    else:

        st.info(
            forecast["message"]
        )


# ---------------------------------------------------------
# Anomaly
# ---------------------------------------------------------

with ai_col2:

    st.markdown("### Consumption anomaly")

    if anomaly["available"]:

        if anomaly["is_anomaly"]:

            st.error(
                "Unusual consumption detected."
            )

        else:

            st.success(
                "Consumption is within the normal range."
            )

        anomaly_col1, anomaly_col2 = st.columns(2)

        with anomaly_col1:

            st.metric(
                "Anomaly score",
                f"{anomaly['score']:.2f}",
            )

        with anomaly_col2:

            st.metric(
                "Live baseline",
                f"{anomaly['baseline_kw']:.2f} kW",
            )

    else:

        st.info(
            anomaly["message"]
        )


# ---------------------------------------------------------
# Recommendations
# ---------------------------------------------------------

st.markdown(
    '<div class="section-title">Data-driven recommendations</div>',
    unsafe_allow_html=True,
)

if recommendations:

    for recommendation in recommendations[:5]:

        st.markdown(
            f"""
            <div class="recommendation-card">
                {recommendation}
            </div>
            """,
            unsafe_allow_html=True,
        )

else:

    st.success(
        "No immediate energy-saving action is required."
    )


# ---------------------------------------------------------
# Live consumption history
# ---------------------------------------------------------

st.markdown(
    '<div class="section-title">Live consumption trend</div>',
    unsafe_allow_html=True,
)

st.session_state.power_history.append(
    energy["power_kw"]
)

st.session_state.timestamps.append(
    energy["timestamp"]
)

# Keep the last 60 observations.
st.session_state.power_history = (
    st.session_state.power_history[-60:]
)

st.session_state.timestamps = (
    st.session_state.timestamps[-60:]
)

if len(st.session_state.power_history) > 1:

    chart_df = pd.DataFrame(
        {
            "Timestamp": pd.to_datetime(
                st.session_state.timestamps
            ),
            "Power (kW)": (
                st.session_state.power_history
            ),
        }
    )

    chart_df = chart_df.set_index(
        "Timestamp"
    )

    st.line_chart(
        chart_df,
        height=350,
    )

else:

    st.info(
        "Collecting live measurements..."
    )


# ---------------------------------------------------------
# Footer / timestamp
# ---------------------------------------------------------

st.divider()

st.markdown(
    f"""
    <div class="small-text">
        Last update: {energy['timestamp']} |
        Data source: live weather API + household simulator |
        Forecast source: trained historical-data model
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# Automatic refresh
# ---------------------------------------------------------

if st.session_state.auto_refresh:

    time.sleep(
        st.session_state.refresh_seconds
    )

    st.rerun()
