import streamlit as st
import pandas as pd
from pathlib import Path
import json
import plotly.express as px

st.set_page_config(page_title="EnergySavvy AI", layout="wide")

# Paths - Colab-safe
BASE = Path(__file__).parent.parent
DATA_HOURLY = BASE / "data/household_power/processed/household_power_hourly.parquet"
DATA_DAILY = BASE / "data/household_power/processed/household_power_daily.csv"
REC_CSV = BASE / "reports/results/recommendations.csv"
REC_SUM = BASE / "reports/results/recommendation_summary.json"
REC_DASH = BASE / "dashboard/data/recommendations.json"
CAIRO_DASH = BASE / "dashboard/data/cairo_weather.json"
CAIRO_SUM = BASE / "reports/results/cairo_weather_summary.json"

st.title("⚡ EnergySavvy AI - Household Energy Intelligence")

tab1, tab2, tab3, tab4 = st.tabs(["🏠 Household EDA", "🔮 Forecasting & Anomaly", "💡 Recommendations", "🌡️ Cairo Weather"])

with tab1:
    st.header("Household Power Consumption")
    if DATA_DAILY.exists():
        df_d = pd.read_csv(DATA_DAILY, parse_dates=['datetime'])
        st.metric("Daily Records", len(df_d))
        fig = px.line(df_d.tail(100), x='datetime', y='Global_active_power', title="Last 100 Days - Active Power")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(df_d.tail())
    else:
        st.warning("Upload household_power_daily.csv to data/household_power/processed/")

with tab2:
    st.header("Anomaly Detection (from 04)")
    anom_path = BASE / "reports/results/anomaly_report.csv"
    if anom_path.exists():
        df_anom = pd.read_csv(anom_path)
        st.metric("Anomalies Found", len(df_anom))
        st.dataframe(df_anom.head(20))
        fig = px.scatter(df_anom, x='datetime', y='Global_active_power', color='anomaly_score', title="Anomalies")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Upload anomaly_report.csv")

with tab3:
    st.header("Recommendation System (from 05)")
    if REC_SUM.exists():
        with open(REC_SUM) as f:
            summary = json.load(f)
        st.json(summary)
    if REC_CSV.exists():
        df_rec = pd.read_csv(REC_CSV)
        st.dataframe(df_rec)
        # Show figures if exist
        for img in ["recommendation_peak_hours.png", "recommendation_types.png"]:
            p = BASE / "reports/figures" / img
            if p.exists():
                st.image(str(p))
    else:
        st.warning("Upload recommendations.csv from 05")

with tab4:
    st.header("Cairo Weather - Independent EDA (from 06)")
    st.info("⚠️ Cairo weather is independent - NOT merged with UCI household (France vs Egypt). Shown as contextual tab.")
    if CAIRO_SUM.exists():
        with open(CAIRO_SUM) as f:
            cairo_summary = json.load(f)
        col1, col2, col3 = st.columns(3)
        col1.metric("Mean Temp", f"{cairo_summary['temp_stats']['mean']:.1f} °C")
        col2.metric("Max Temp", f"{cairo_summary['temp_stats']['max']:.1f} °C")
        col3.metric("Min Temp", f"{cairo_summary['temp_stats']['min']:.1f} °C")
        st.json(cairo_summary)

        # Monthly chart from summary
        monthly = cairo_summary.get('monthly_avg_temp', {})
        if monthly:
            df_m = pd.DataFrame(list(monthly.items()), columns=['month','temp'])
            fig = px.bar(df_m, x='month', y='temp', title="Cairo Monthly Avg Temp - Seasonality")
            st.plotly_chart(fig, use_container_width=True)

    if CAIRO_DASH.exists():
        st.subheader("Last 30 records - Cairo")
        with open(CAIRO_DASH) as f:
            dash_data = json.load(f)
        df_sample = pd.DataFrame(dash_data.get('sample_last_30', []))
        if not df_sample.empty:
            st.dataframe(df_sample)

    # Show cairo figures
    for img in ["cairo_weather_monthly_temp.png", "cairo_weather_yearly_temp.png", "cairo_weather_temp_hist.png", "cairo_weather_temp_boxplot.png"]:
        p = BASE / "reports/figures" / img
        if p.exists():
            st.image(str(p), caption=img)

st.sidebar.success("All notebooks 01-06 completed. Dashboard ready.")
st.sidebar.markdown("[GitHub Repo](https://github.com/Mohamed-Al-Saudi/EnergySavvy-AI-Project)")
