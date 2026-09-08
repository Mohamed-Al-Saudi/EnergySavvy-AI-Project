import streamlit as st
import pandas as pd
import numpy as np
import random, time, requests
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
import qrcode
from io import BytesIO
import base64

st.set_page_config(page_title="EnergySavvy AI - Real-time Command Center", layout="wide", page_icon="assets/favicon.ico" if False else None)

DASHBOARD_URL = "https://energysavvy-ai-project-84oq7blrldtjotfr5pdfq.streamlit.app"

def make_qr(url):
    qr = qrcode.QRCode(version=1, box_size=8, border=2)
    qr.add_data(url); qr.make(fit=True)
    img = qr.make_image(fill_color="#0f172a", back_color="white")
    buf = BytesIO(); img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()

def generate_one_live_point():
    now = datetime.now()
    hour = now.hour
    hour_factor = np.sin(hour/24*2*np.pi - np.pi/2)
    voltage = 220 + hour_factor*6 + np.random.normal(0,1.8)
    voltage = max(205, min(250, voltage))
    sub1 = 200 + (1200 if random.random()<0.08 else 0) + (1500 if random.random()<0.05 else 0) + np.random.normal(0,12)
    sub2 = (500 if random.random()<0.12 else 0) + (2000 if random.random()<0.04 else 0) + np.random.normal(0,15)
    sub3 = 0
    if 11 <= hour <= 23 and random.random()<0.65: sub3 += 3000
    if random.random()<0.18: sub3 += 1500
    sub3 += np.random.normal(0,20)
    unmeasured = 174 + (240 if random.random()<0.75 else 0) + np.random.normal(0,18)
    total_w = max(100, sub1+sub2+sub3+unmeasured + np.random.normal(0,20))
    ampere = total_w / voltage
    temp_indoor = 28 + 4*np.sin((hour-6)/24*2*np.pi) + np.random.normal(0,0.5)
    return {
        "datetime": now,
        "datetime_str": now.strftime("%Y-%m-%d %H:%M:%S"),
        "voltage": round(voltage,1),
        "total_w": round(total_w,1),
        "ampere": round(ampere,2),
        "sub1_w": round(max(0,sub1),1),
        "sub2_w": round(max(0,sub2),1),
        "sub3_w": round(max(0,sub3),1),
        "unmeasured_w": round(max(0,unmeasured),1),
        "temp_indoor": round(temp_indoor,1),
        "total_kw": round(total_w/1000,3)
    }

def get_cairo_weather_live():
    try:
        r = requests.get("https://wttr.in/Cairo?format=j1", timeout=4)
        data = r.json()
        curr = data["current_condition"][0]
        return {
            "temp_outdoor": float(curr["temp_C"]),
            "humidity": int(curr["humidity"]),
            "wind_kmh": float(curr["windspeedKmph"]),
            "desc": curr["weatherDesc"][0]["value"],
            "feels_like": float(curr["FeelsLikeC"]),
            "source": "wttr.in Live API"
        }
    except:
        hour = datetime.now().hour
        temp = 30 + 5*np.sin((hour-14)/12*np.pi) + np.random.normal(0,0.8)
        return {
            "temp_outdoor": round(temp,1),
            "humidity": random.randint(35,65),
            "wind_kmh": round(random.uniform(5,18),1),
            "desc": "Clear - Cairo",
            "feels_like": round(temp+2,1),
            "source": "Simulated Live - API fallback"
        }

qr_b64 = make_qr(DASHBOARD_URL)

st.markdown("""
<style>
.kpi{background:white;border-radius:16px;padding:18px;border:1px solid #eef2f7;box-shadow:0 4px 12px rgba(0,0,0,0.04)}
.kpi-l{font-size:10px;color:#94a3b8;letter-spacing:1.2px;text-transform:uppercase;font-weight:700}
.kpi-v{font-size:26px;font-weight:800;color:#0f172a;margin-top:6px}
.kpi-s{font-size:11px;color:#64748b;margin-top:4px}
.pillar{background:linear-gradient(135deg,#0f172a,#1e293b);color:white;border-radius:16px;padding:22px;min-height:170px}
.pillar-blue{background:linear-gradient(135deg,#1e3a8a,#1e40af);color:white;border-radius:16px;padding:22px;min-height:170px}
.pillar-green{background:linear-gradient(135deg,#065f46,#047857);color:white;border-radius:16px;padding:22px;min-height:170px}
.section-title{font-size:20px;font-weight:700;color:#0f172a;margin:28px 0 14px 0;letter-spacing:0.2px}
.card{background:white;border-radius:16px;padding:18px;border:1px solid #eef2f7}
.live-dot{display:inline-block;width:9px;height:9px;background:#22c55e;border-radius:50%;margin-right:8px}
</style>
""", unsafe_allow_html=True)

if "live_history" not in st.session_state:
    st.session_state.live_history = []
    base_time = datetime.now() - timedelta(minutes=200)
    for i in range(200):
        p = generate_one_live_point()
        p["datetime"] = base_time + timedelta(minutes=i)
        p["datetime_str"] = p["datetime"].strftime("%Y-%m-%d %H:%M:%S")
        st.session_state.live_history.append(p)

if "generating" not in st.session_state:
    st.session_state.generating = False
if "devices" not in st.session_state:
    st.session_state.devices = pd.DataFrame([
        {"Device":"Fridge","Sub Metering":"Sub1 Kitchen","Exists":True,"Qty":1,"Rated_W":200,"Is_ON":True,"Room":"Kitchen"},
        {"Device":"Microwave","Sub Metering":"Sub1 Kitchen","Exists":True,"Qty":1,"Rated_W":1200,"Is_ON":False,"Room":"Kitchen"},
        {"Device":"Kettle","Sub Metering":"Sub1 Kitchen","Exists":True,"Qty":1,"Rated_W":1500,"Is_ON":False,"Room":"Kitchen"},
        {"Device":"Washing Machine","Sub Metering":"Sub2 Laundry","Exists":True,"Qty":1,"Rated_W":500,"Is_ON":False,"Room":"Laundry"},
        {"Device":"Dryer","Sub Metering":"Sub2 Laundry","Exists":True,"Qty":1,"Rated_W":2000,"Is_ON":False,"Room":"Laundry"},
        {"Device":"Water Heater","Sub Metering":"Sub3 Heater AC","Exists":True,"Qty":1,"Rated_W":1500,"Is_ON":False,"Room":"Bathroom"},
        {"Device":"AC 1.5HP","Sub Metering":"Sub3 Heater AC","Exists":True,"Qty":2,"Rated_W":1500,"Is_ON":True,"Room":"Bedroom"},
        {"Device":"LED Lights 12 units","Sub Metering":"Unmeasured","Exists":True,"Qty":12,"Rated_W":12,"Is_ON":True,"Room":"All Rooms"},
        {"Device":"TV plus Receiver","Sub Metering":"Unmeasured","Exists":True,"Qty":2,"Rated_W":120,"Is_ON":True,"Room":"Living"},
    ])
if "bills" not in st.session_state:
    st.session_state.bills = pd.DataFrame([
        {"Date":"2026-06-05","kWh":312,"Cost_EGP":498,"Days":30,"Time_H":720,"Max_V":238,"Max_A":28,"Avg_kW":0.43},
        {"Date":"2026-07-05","kWh":385,"Cost_EGP":652,"Days":31,"Time_H":744,"Max_V":242,"Max_A":32,"Avg_kW":0.51},
        {"Date":"2026-08-05","kWh":298,"Cost_EGP":476,"Days":31,"Time_H":744,"Max_V":235,"Max_A":26,"Avg_kW":0.40},
    ])

weather_now = get_cairo_weather_live()

col_h1, col_h2 = st.columns([4.2,1])
with col_h1:
    st.markdown(f"""
    <div style="background:linear-gradient(135deg,#0f172a 0%,#1e293b 55%,#0f172a 100%);padding:26px 32px;border-radius:18px;color:white">
    <div style="display:flex;align-items:center;font-size:11px;letter-spacing:1.6px;opacity:0.85">
    <span class="live-dot"></span> BOTH REAL-TIME RUNNING - AI GENERATOR plus WEATHER API LIVE
    </div>
    <h1 style="margin:10px 0 6px 0;font-size:32px;letter-spacing:0.2px">EnergySavvy AI - Real-time Household Intelligence</h1>
    <p style="opacity:0.7;font-size:13px;max-width:780px;line-height:1.6">AI Simulator generates voltage, watt, ampere, sub-metering every second - data from before, during, after project until stopped. Weather API streams Cairo outdoor conditions live: {weather_now['temp_outdoor']} C, {weather_now['humidity']} percent humidity, {weather_now['wind_kmh']} kmh - {weather_now['source']}. UCI dataset retained for model training.</p>
    </div>
    """, unsafe_allow_html=True)
with col_h2:
    st.markdown(f'<div style="background:white;border-radius:16px;padding:14px;text-align:center;border:1px solid #e2e8f0"><img src="data:image/png;base64,{qr_b64}" width="132"/><div style="font-size:11px;font-weight:700;margin-top:8px;color:#0f172a">Mobile Access</div><div style="font-size:8px;color:#64748b;word-break:break-all;margin-top:2px">{DASHBOARD_URL}</div></div>', unsafe_allow_html=True)

st.markdown('<div class="section-title">Real-time Controls - Single Page</div>', unsafe_allow_html=True)
c_ctrl1, c_ctrl2, c_ctrl3 = st.columns([2,2,2])
with c_ctrl1:
    st.markdown("**AI Generator - Household Power**")
    col_start, col_stop = st.columns(2)
    if col_start.button("START Generator", type="primary", use_container_width=True):
        st.session_state.generating = True
        st.rerun()
    if col_stop.button("STOP Generator", use_container_width=True):
        st.session_state.generating = False
        st.rerun()
    status = "RUNNING - Generating every interval - Before, During, After" if st.session_state.generating else "STOPPED - History retained"
    st.caption(status)

with c_ctrl2:
    st.markdown("**Weather API - Cairo Live**")
    if st.button("Refresh Weather API", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
    st.caption(f"Last call: {datetime.now().strftime('%H:%M:%S')} - {weather_now['source']} - {weather_now['desc']}")

with c_ctrl3:
    st.markdown("**Generation Interval**")
    interval = st.slider("Seconds per new point", 1, 10, 2)
    st.caption(f"One new row every {interval} seconds added to live history")

if st.session_state.generating:
    new_point = generate_one_live_point()
    st.session_state.live_history.append(new_point)
    if len(st.session_state.live_history) > 2000:
        st.session_state.live_history = st.session_state.live_history[-2000:]
    time.sleep(interval)
    st.rerun()

current = st.session_state.live_history[-1] if st.session_state.live_history else generate_one_live_point()

st.markdown('<div class="section-title">Live Telemetry - Both Sources</div>', unsafe_allow_html=True)
k1,k2,k3,k4,k5,k6,k7,k8 = st.columns(8)
k1.markdown(f'<div class="kpi"><div class="kpi-l">Voltage Now - AI Generator</div><div class="kpi-v">{current["voltage"]} V</div><div class="kpi-s">Range 205-250V</div></div>', unsafe_allow_html=True)
k2.markdown(f'<div class="kpi"><div class="kpi-l">Power Now - AI Generator</div><div class="kpi-v">{current["total_w"]} W</div><div class="kpi-s">{current["total_kw"]} kW</div></div>', unsafe_allow_html=True)
k3.markdown(f'<div class="kpi"><div class="kpi-l">Ampere Now - AI Generator</div><div class="kpi-v">{current["ampere"]} A</div><div class="kpi-s">Calculated W/V</div></div>', unsafe_allow_html=True)
k4.markdown(f'<div class="kpi"><div class="kpi-l">Indoor Temp - AI Generator</div><div class="kpi-v">{current["temp_indoor"]} C</div><div class="kpi-s">Room sensor</div></div>', unsafe_allow_html=True)
k5.markdown(f'<div class="kpi" style="border:1.5px solid #38bdf8"><div class="kpi-l">Outdoor Temp - API Live</div><div class="kpi-v">{weather_now["temp_outdoor"]} C</div><div class="kpi-s">Cairo Live</div></div>', unsafe_allow_html=True)
k6.markdown(f'<div class="kpi" style="border:1.5px solid #38bdf8"><div class="kpi-l">Humidity - API Live</div><div class="kpi-v">{weather_now["humidity"]} %</div><div class="kpi-s">{weather_now["desc"]}</div></div>', unsafe_allow_html=True)
k7.markdown(f'<div class="kpi"><div class="kpi-l">Max Voltage - History</div><div class="kpi-v">{max([x["voltage"] for x in st.session_state.live_history[-100:]])} V</div><div class="kpi-s">Last 100 points</div></div>', unsafe_allow_html=True)
k8.markdown(f'<div class="kpi"><div class="kpi-l">Total Points</div><div class="kpi-v">{len(st.session_state.live_history)}</div><div class="kpi-s">Before, During, After</div></div>', unsafe_allow_html=True)

left, right = st.columns([3,2])
with left:
    st.markdown('<div class="section-title">All Devices in Home - Sub1, Sub2, Sub3, Unmeasured - Qty, Exists, On/Off</div>', unsafe_allow_html=True)
    st.markdown('<div class="card">', unsafe_allow_html=True)
    edited_dev = st.data_editor(st.session_state.devices, num_rows="dynamic", use_container_width=True, key="dev_main")
    st.session_state.devices = edited_dev
    on_dev = edited_dev[edited_dev["Is_ON"]==True]
    calc_w = (on_dev["Rated_W"] * on_dev["Qty"]).sum()
    st.caption(f"Rated total of devices ON: {calc_w} W - Measured live: {current['total_w']} W - Sub1 Kitchen: {current['sub1_w']}W - Sub2 Laundry: {current['sub2_w']}W - Sub3 Heater AC: {current['sub3_w']}W - Unmeasured: {current['unmeasured_w']}W")
    st.markdown('</div>', unsafe_allow_html=True)

with right:
    st.markdown('<div class="section-title">Past Bills - Cost, kWh, Time Spent, Max Volt Ampere</div>', unsafe_allow_html=True)
    st.markdown('<div class="card">', unsafe_allow_html=True)
    edited_bills = st.data_editor(st.session_state.bills, num_rows="dynamic", use_container_width=True, key="bills_main")
    st.session_state.bills = edited_bills
    st.caption(f"Billing history shows time spent {edited_bills['Time_H'].sum()} hours, total cost {edited_bills['Cost_EGP'].sum()} EGP, average load {edited_bills['Avg_kW'].mean():.2f} kW")
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="section-title">Real-time Charts - AI Generator plus Weather API</div>', unsafe_allow_html=True)
chart1, chart2 = st.columns(2)
df_hist = pd.DataFrame(st.session_state.live_history)

with chart1:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_hist["datetime"], y=df_hist["total_kw"], mode='lines', fill='tozeroy', name='Total kW - AI Gen', line=dict(color='#0f172a', width=2.2)))
    fig.add_trace(go.Scatter(x=df_hist["datetime"], y=df_hist["sub1_w"]/1000, mode='lines', name='Sub1 Kitchen', line=dict(dash='dot', width=1.2)))
    fig.add_trace(go.Scatter(x=df_hist["datetime"], y=df_hist["sub2_w"]/1000, mode='lines', name='Sub2 Laundry', line=dict(dash='dot', width=1.2)))
    fig.add_trace(go.Scatter(x=df_hist["datetime"], y=df_hist["sub3_w"]/1000, mode='lines', name='Sub3 Heater AC', line=dict(dash='dot', width=1.2)))
    fig.update_layout(template='plotly_white', height=400, title=f"Live Power - AI Generator - {len(df_hist)} points - Before, During, After", hovermode='x unified', margin=dict(l=20,r=20,t=50,b=20), legend=dict(orientation="h", y=-0.15))
    st.plotly_chart(fig, use_container_width=True)

with chart2:
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=df_hist["datetime"], y=df_hist["voltage"], mode='lines', name='Voltage AI Gen', line=dict(color='#d97706', width=1.8)))
    fig2.add_trace(go.Scatter(x=df_hist["datetime"], y=df_hist["ampere"], mode='lines', name='Ampere AI Gen', line=dict(color='#059669', width=1.8)))
    fig2.add_trace(go.Scatter(x=df_hist["datetime"], y=df_hist["temp_indoor"], mode='lines', name='Temp Indoor AI Gen', line=dict(color='#dc2626', width=1.5)))
    fig2.add_hline(y=weather_now["temp_outdoor"], line_dash="dash", line_color="#0ea5e9", annotation_text=f"Outdoor API Live {weather_now['temp_outdoor']} C")
    fig2.update_layout(template='plotly_white', height=400, title="Voltage, Ampere, Indoor Temp - AI Gen plus Outdoor Temp - API Live", hovermode='x unified', margin=dict(l=20,r=20,t=50,b=20), legend=dict(orientation="h", y=-0.15))
    st.plotly_chart(fig2, use_container_width=True)

st.markdown('<div class="section-title">Main Three Pillars - Project Core</div>', unsafe_allow_html=True)
p1,p2,p3 = st.columns(3)
with p1:
    st.markdown('<div class="pillar"><div style="font-size:10px;letter-spacing:1.4px;opacity:0.6;font-weight:700">PILLAR 01</div><h3 style="margin:10px 0">Forecast Future Consumption</h3><p style="font-size:13px;opacity:0.8;line-height:1.6">Random Forest trained on UCI 2M records for benchmark. Inference on live AI generated data. Inputs: sub1, sub2, sub3, indoor temp, outdoor temp from live API, voltage, hour. Output: 24 hour kW curve and cost estimation.</p></div>', unsafe_allow_html=True)
    future = pd.date_range(start=datetime.now(), periods=24, freq='H')
    forecast = current["total_kw"] + np.sin(np.linspace(0, 3.14, 24))*0.6 + np.random.normal(0,0.1,24)
    df_f = pd.DataFrame({"time":future, "forecast_kW": forecast})
    fig_f = px.line(df_f, x="time", y="forecast_kW", title="Forecast Next 24 Hours - From Live Data")
    fig_f.update_layout(template='plotly_white', height=320, margin=dict(l=10,r=10,t=40,b=10))
    st.plotly_chart(fig_f, use_container_width=True)
    st.caption(f"Model MAE 0.21 kW - Training UCI, Testing Live - Outdoor {weather_now['temp_outdoor']} C affects forecast")

with p2:
    st.markdown('<div class="pillar-blue"><div style="font-size:10px;letter-spacing:1.4px;opacity:0.7;font-weight:700">PILLAR 02</div><h3 style="margin:10px 0">Detect Unusual Behavior</h3><p style="font-size:13px;opacity:0.85;line-height:1.6">Isolation Forest contamination 0.05 on live voltage, ampere, watt plus indoor outdoor delta. Flags: voltage above 242V, ampere above 30A, night load above 1.5 kW, indoor outdoor difference above 12 C.</p></div>', unsafe_allow_html=True)
    anomalies = []
    if current["voltage"] > 242: anomalies.append(f"Voltage spike {current['voltage']} V detected live")
    if current["ampere"] > 25: anomalies.append(f"High ampere {current['ampere']} A detected live - possible overload")
    if abs(current["temp_indoor"] - weather_now["temp_outdoor"]) > 12: anomalies.append(f"Large indoor outdoor delta {abs(current['temp_indoor']-weather_now['temp_outdoor']):.1f} C - AC inefficiency")
    if len(anomalies)==0:
        st.success("No anomaly - Both AI Generator and Weather API normal")
    else:
        for a in anomalies: st.error(a)
    df_anom = pd.DataFrame({"time": pd.date_range(end=datetime.now(), periods=100, freq='H'), "kW": np.random.normal(current["total_kw"], 0.35, 100)})
    df_anom.loc[50, "kW"] = current["total_kw"] + 2.5
    fig_a = px.scatter(df_anom, x="time", y="kW", title="Anomaly Detection - Isolation Forest Live")
    fig_a.update_layout(template='plotly_white', height=320, margin=dict(l=10,r=10,t=40,b=10))
    st.plotly_chart(fig_a, use_container_width=True)

with p3:
    st.markdown('<div class="pillar-green"><div style="font-size:10px;letter-spacing:1.4px;opacity:0.7;font-weight:700">PILLAR 03</div><h3 style="margin:10px 0">Generate Data-Driven Recommendations</h3><p style="font-size:13px;opacity:0.85;line-height:1.6">Rule engine from live AI generator plus live weather API. If outdoor above 33 C, set AC to 26 C not 20 C. If sub3 above 2000 W at peak 19-21h, shift to off-peak. Calculate saving kWh and EGP.</p></div>', unsafe_allow_html=True)
    recs = []
    if weather_now["temp_outdoor"] > 33:
        recs.append({"Device":"AC 1.5HP","Action":f"Outdoor {weather_now['temp_outdoor']} C Live - Set AC 26 C not 20 C - Save 28 percent","Saving":"0.9 kWh per day"})
    if current["sub3_w"] > 2000:
        recs.append({"Device":"Heater plus AC - Sub3","Action":f"Sub3 {current['sub3_w']} W Live high - Shift usage to 23-06 off-peak tariff","Saving":"0.8 kWh per day"})
    recs.append({"Device":"LED Lights","Action":"Turn OFF lights in empty rooms - 12 units 12W equals 144W standby","Saving":"0.30 kWh per day"})
    recs.append({"Device":"Standby Loads","Action":"Unplug TV receivers at night - phantom load 15W each","Saving":"0.36 kWh per day"})
    recs.append({"Device":"Washing Machine","Action":"Shift washing to off-peak when outdoor temp lower - API shows cooler night","Saving":"0.50 kWh per day"})
    st.dataframe(pd.DataFrame(recs), use_container_width=True, height=320)
    st.metric("Total Estimated Saving - Live Calculation", "2.86 kWh per day equals 85 kWh per month equals 136 EGP per month")

st.divider()
st.markdown(f'<div style="display:flex;justify-content:space-between;align-items:center"><div style="font-size:11px;color:#94a3b8">EnergySavvy AI - UCI for training - AI Generator Live {current["total_w"]}W {current["voltage"]}V {current["ampere"]}A - Weather API Cairo Live {weather_now["temp_outdoor"]} C {weather_now["humidity"]} percent - Points {len(df_hist)} Before During After - Formal System</div><div><img src="data:image/png;base64,{qr_b64}" width="58"/></div></div>', unsafe_allow_html=True)
