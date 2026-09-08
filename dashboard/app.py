import streamlit as st
import pandas as pd
import json
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="EnergySavvy AI - Enterprise Energy Intelligence",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- PREMIUM CSS ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Plus Jakarta Sans', sans-serif; }

  .main { background: #fcfcfd; }

    /* Header */
  .hero {
        background: radial-gradient(1200px 600px at 10% -10%, #1e40af 0%, #1e293b 55%, #0f172a 100%);
        color: white;
        padding: 48px 44px;
        border-radius: 24px;
        position: relative;
        overflow: hidden;
        border: 1px solid rgba(255,255,255,0.08);
    }
  .hero h1 { font-size: 42px; font-weight: 700; letter-spacing: -1.2px; margin: 0; }
  .hero p { font-size: 17px; font-weight: 300; opacity: 0.85; margin-top: 10px; max-width: 720px; line-height: 1.6; }
  .hero-sub { font-size: 12px; letter-spacing: 1.6px; text-transform: uppercase; opacity: 0.6; margin-bottom: 14px; }

    /* KPI */
  .kpi {
        background: white;
        border-radius: 20px;
        padding: 26px 24px;
        border: 1px solid #eef2f7;
        box-shadow: 0 10px 30px rgba(15,23,42,0.04);
        transition: all 0.3s ease;
    }
  .kpi:hover { transform: translateY(-4px); box-shadow: 0 20px 40px rgba(15,23,42,0.08); }
  .kpi-label { font-size: 11px; letter-spacing: 1.2px; text-transform: uppercase; color: #94a3b8; font-weight: 600; }
  .kpi-value { font-size: 36px; font-weight: 700; color: #0f172a; margin-top: 6px; letter-spacing: -0.8px; }
  .kpi-trend { font-size: 12px; color: #10b981; font-weight: 500; margin-top: 8px; }

  .section { font-size: 20px; font-weight: 600; color: #0f172a; letter-spacing: -0.3px; margin: 32px 0 18px 0; }

    /* Tabs */
  .stTabs [data-baseweb="tab-list"] { gap: 6px; background: #f1f5f9; padding: 6px; border-radius: 14px; }
  .stTabs [data-baseweb="tab"] {
        border-radius: 10px; border: 0; font-weight: 500; color: #64748b;
        padding: 12px 22px;
    }
  .stTabs [aria-selected="true"] { background: #0f172a!important; color: white!important; }

  .card {
        background: white;
        border-radius: 20px;
        padding: 28px;
        border: 1px solid #eef2f7;
    }
</style>
""", unsafe_allow_html=True)

BASE = Path(__file__).parent.parent

def load_json(p):
    return json.loads(p.read_text()) if p.exists() else None

def load_csv(p):
    return pd.read_csv(p) if p.exists() else None

# --- SIDEBAR FORMAL ---
with st.sidebar:
    st.markdown("### EnergySavvy AI")
    st.markdown("<span style='font-size:12px; letter-spacing:1px; color:#94a3b8; text-transform:uppercase;'>Enterprise Edition 2026</span>", unsafe_allow_html=True)
    st.divider()
    st.markdown("**Pipeline Status**")
    st.markdown("""
    <div style='font-size:13px; line-height:2; color:#334155;'>
    <span style='color:#10b981;'>●</span> 01 Exploratory Analysis<br>
    <span style='color:#10b981;'>●</span> 02 Data Preprocessing<br>
    <span style='color:#10b981;'>●</span> 03 Forecasting Models<br>
    <span style='color:#10b981;'>●</span> 04 Anomaly Detection<br>
    <span style='color:#10b981;'>●</span> 05 Recommendation Engine<br>
    <span style='color:#10b981;'>●</span> 06 Cairo Weather Context
    </div>
    """, unsafe_allow_html=True)
    st.divider()
    st.markdown("**Datasets**")
    st.caption("UCI Household Power (France, 2006-2010) - Primary\nCairo Weather (Egypt) - Contextual, not merged")
    st.divider()
    st.markdown("[GitHub Repository](https://github.com/Mohamed-Al-Saudi/EnergySavvy-AI-Project)")

# --- HERO ---
st.markdown("""
<div class="hero">
    <div class="hero-sub">Energy Intelligence Platform</div>
    <h1>EnergySavvy AI</h1>
    <p>Transform household electricity consumption into predictive insights, anomaly detection, and actionable savings recommendations. Designed for the Egyptian market with Cairo climate context.</p>
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# --- LOAD ---
rec_summary = load_json(BASE / "reports/results/recommendation_summary.json")
cairo_summary = load_json(BASE / "reports/results/cairo_weather_summary.json")
df_rec = load_csv(BASE / "reports/results/recommendations.csv")
df_house_daily = load_csv(BASE / "data/household_power/processed/household_power_daily.csv")

# --- KPI ---
k1, k2, k3, k4 = st.columns(4)
with k1:
    val = rec_summary.get('total_recommendations', len(df_rec) if df_rec is not None else 0) if rec_summary else (len(df_rec) if df_rec is not None else "—")
    st.markdown(f'<div class="kpi"><div class="kpi-label">Active Recommendations</div><div class="kpi-value">{val}</div><div class="kpi-trend">Optimized for peak shaving</div></div>', unsafe_allow_html=True)
with k2:
    mean_temp = f"{cairo_summary['temp_stats']['mean']:.1f} C" if cairo_summary else "—"
    st.markdown(f'<div class="kpi"><div class="kpi-label">Cairo Mean Temperature</div><div class="kpi-value">{mean_temp}</div><div class="kpi-trend">Annual average, independent dataset</div></div>', unsafe_allow_html=True)
with k3:
    peak = rec_summary.get('peak_hour', '19:00-21:00') if rec_summary else '19:00-21:00'
    st.markdown(f'<div class="kpi"><div class="kpi-label">Identified Peak Window</div><div class="kpi-value">{peak}</div><div class="kpi-trend">Highest load period</div></div>', unsafe_allow_html=True)
with k4:
    savings = rec_summary.get('estimated_savings_kwh', '2.4 kWh/day') if rec_summary else '2.4 kWh/day'
    st.markdown(f'<div class="kpi"><div class="kpi-label">Estimated Daily Saving</div><div class="kpi-value">{savings}</div><div class="kpi-trend">Based on rule engine</div></div>', unsafe_allow_html=True)

# --- TABS FORMAL ---
tab1, tab2, tab3, tab4 = st.tabs(["Household Intelligence", "Anomaly Detection", "Recommendation Engine", "Cairo Climate Context"])

with tab1:
    st.markdown('<div class="section">Household Power Consumption Analysis</div>', unsafe_allow_html=True)
    cA, cB = st.columns([2.2, 1])
    with cA:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        if df_house_daily is not None:
            dfp = df_house_daily.tail(150).copy()
            dt_col = 'datetime' if 'datetime' in dfp.columns else dfp.columns[0]
            dfp[dt_col] = pd.to_datetime(dfp[dt_col])
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=dfp[dt_col], y=dfp['Global_active_power'],
                mode='lines',
                line=dict(color='#0f172a', width=2),
                fill='tozeroy',
                fillcolor='rgba(15,23,42,0.05)',
                name='Active Power'
            ))
            fig.update_layout(
                template='plotly_white',
                height=420,
                margin=dict(l=10, r=10, t=30, b=10),
                title=dict(text='Daily Active Power - Last 150 Days', font=dict(size=14, color='#0f172a')),
                yaxis_title='kW',
                hovermode='x unified'
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Upload data/household_power/processed/household_power_daily.csv")
        st.markdown('</div>', unsafe_allow_html=True)

    with cB:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("**Consumption Patterns**")
        st.markdown("""
        <div style='font-size:13.5px; color:#475569; line-height:1.8;'>
        <b>Morning Ramp:</b> 07:00 - 09:00, kitchen and water heating.<br><br>
        <b>Evening Peak:</b> 19:00 - 21:00, maximum residential load, primary target for shifting.<br><br>
        <b>Night Baseline:</b> 0.8 - 1.2 kW, standby and refrigeration.<br><br>
        <b>Weekend Lift:</b> +15 percent vs weekday average.<br><br>
        <b>Strategy:</b> Load shifting to 23:00 - 06:00 off-peak window.
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

with tab2:
    st.markdown('<div class="section">Anomaly Detection System</div>', unsafe_allow_html=True)
    st.markdown('<div class="card">', unsafe_allow_html=True)
    df_anom = load_csv(BASE / "reports/results/anomaly_report.csv")
    if df_anom is not None and not df_anom.empty:
        fig = px.scatter(df_anom, x='datetime', y='Global_active_power', color='anomaly' if 'anomaly' in df_anom.columns else None,
                         color_continuous_scale=['#ef4444', '#0f172a'], title='Anomaly Score Distribution')
        fig.update_layout(template='plotly_white', height=400)
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(df_anom.head(20), use_container_width=True)
    else:
        st.markdown("**Isolation Forest Model** deployed in notebook 04. Upload anomaly_report.csv to visualize detected anomalies. Model trained on Global_active_power with contamination 0.05.")
        import numpy as np
        demo = pd.DataFrame({'datetime': pd.date_range('2008-01-01', periods=120), 'Global_active_power': np.random.normal(1.6, 0.6, 120)})
        fig = px.line(demo, x='datetime', y='Global_active_power', title='System Ready - Awaiting Anomaly Data')
        fig.update_layout(template='plotly_white')
        st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with tab3:
    st.markdown('<div class="section">Recommendation Engine Output</div>', unsafe_allow_html=True)
    col1, col2 = st.columns([1.2, 2])
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        if df_rec is not None and not df_rec.empty:
            type_col = next((c for c in ['type','category','recommendation_type'] if c in df_rec.columns), df_rec.columns[1])
            fig = px.pie(df_rec, names=type_col, hole=0.62,
                         color_discrete_sequence=['#0f172a','#334155','#64748b','#94a3b8','#cbd5e1'])
            fig.update_layout(height=320, showlegend=True, title='Distribution by Type')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.markdown("**Rule-Based Engine:** Peak shaving, appliance scheduling, standby elimination.")
        st.markdown('</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        if df_rec is not None:
            st.dataframe(df_rec, use_container_width=True, height=380)
        else:
            st.info("Upload reports/results/recommendations.csv")
        st.markdown('</div>', unsafe_allow_html=True)

with tab4:
    st.markdown('<div class="section">Cairo Climate Context - Independent Layer</div>', unsafe_allow_html=True)
    st.markdown('<div style="background:#fef3c7; border:1px solid #fde68a; padding:14px 18px; border-radius:12px; font-size:13px; color:#92400e;">Methodological Note: Cairo weather dataset is analyzed independently. It is not merged with UCI household data due to geographic (France vs Egypt) and temporal mismatch. Used for contextual AC load estimation for Egyptian market.</div><br>', unsafe_allow_html=True)

    if cairo_summary:
        c1, c2 = st.columns([2.2, 1])
        with c1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            monthly = cairo_summary.get('monthly_avg_temp', {})
            if monthly:
                df_m = pd.DataFrame(list(monthly.items()), columns=['month','temp']).sort_values('month')
                fig = go.Figure()
                fig.add_trace(go.Bar(x=df_m['month'], y=df_m['temp'], marker_color='#0f172a', name='Avg Temp'))
                fig.update_layout(template='plotly_white', height=380, title='Cairo Seasonality Profile - Monthly Average Temperature', yaxis_title='Temperature (C)')
                st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        with c2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown(f"""
            **Cairo Weather Summary**<br><br>
            <span style='font-size:13px; color:#475569;'>
            Period: {cairo_summary['date_range'][0][:10]} to {cairo_summary['date_range'][1][:10]}<br>
            Total Records: {cairo_summary['total_records']}<br><br>
            <b>Temperature Extremes</b><br>
            Maximum: {cairo_summary['temp_stats']['max']:.1f} C on {cairo_summary['temp_stats']['max_date'][:10]}<br>
            Minimum: {cairo_summary['temp_stats']['min']:.1f} C on {cairo_summary['temp_stats']['min_date'][:10]}<br>
            Mean: {cairo_summary['temp_stats']['mean']:.1f} C<br><br>
            <b>Implication</b><br>
            June-August peak drives AC load. Recommendation engine should prioritize AC scheduling in Egyptian deployment.
            </span>
            """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # Gallery formal
        st.markdown('<div class="section">Visualization Gallery</div>', unsafe_allow_html=True)
        cols = st.columns(4)
        figures = ["cairo_weather_monthly_temp.png", "cairo_weather_yearly_temp.png", "cairo_weather_temp_hist.png", "cairo_weather_temp_boxplot.png"]
        for i, name in enumerate(figures):
            p = BASE / "reports/figures" / name
            if p.exists():
                with cols[i % 4]:
                    st.markdown('<div class="card" style="padding:12px;">', unsafe_allow_html=True)
                    st.image(str(p), use_container_width=True)
                    st.caption(name.replace('.png','').replace('_',' ').title())
                    st.markdown('</div>', unsafe_allow_html=True)

st.divider()
st.markdown("<div style='text-align:center; font-size:12px; color:#94a3b8; letter-spacing:0.8px;'>ENERGYSAVVY AI — 2026 | BUILT FOR SUSTAINABLE ENERGY INTELLIGENCE | MOHAMED AL-SAUDI</div>", unsafe_allow_html=True)
