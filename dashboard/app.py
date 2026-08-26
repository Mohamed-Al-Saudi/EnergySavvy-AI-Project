"""EnergySavvy AI Streamlit entry point."""
import streamlit as st

st.set_page_config(page_title="EnergySavvy AI", page_icon="⚡", layout="wide")

st.title("⚡ EnergySavvy AI")
st.write(
    "A prototype intelligent system for household electricity analysis, "
    "forecasting, anomaly detection, and data-driven recommendations."
)

st.info(
    "Current prototype: UCI household electricity is the primary ML dataset. "
    "Cairo weather is analyzed separately and is not merged with the household data."
)
