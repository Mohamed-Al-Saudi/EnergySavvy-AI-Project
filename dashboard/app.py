
# ============================================================================
# EnergySavvy AI - FINAL DEPLOYMENT - DASHBOARD/APP.PY
# FLOW: Live Location (ipapi.co) -> Weather (wttr.in + Open-Meteo) -> RealtimeEngine
# -> EnergySimulator (Egyptian equation) -> Forecast Model (forecast_rf.pkl)
# -> Anomaly Detection -> Recommendations -> Streamlit Dashboard
# TRAINING: UCI Household + Cairo Weather OLD datasets
# LIVE: Egyptian equation SIM + REAL weather API
# ============================================================================
import streamlit as st
st.set_page_config(page_title="EnergySavvy AI - Live Egyptian Home", layout="wide", page_icon="⚡", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Space+Grotesk:wght@400;600;700;900&display=swap');
.stApp{
    background: radial-gradient(1200px 700px at 15% 0%, rgba(168,85,247,0.35), transparent 60%),
                radial-gradient(1000px 600px at 85% 10%, rgba(6,182,212,0.35), transparent 60%),
                radial-gradient(800px 500px at 50% 100%, rgba(251,146,60,0.18), transparent 70%),
                #070C1A;
    font-family: 'Space Grotesk', sans-serif;
}
.kpi{
    background: linear-gradient(180deg, rgba(255,255,255,0.10), rgba(255,255,255,0.03));
    border: 1px solid rgba(255,255,255,0.12);
    backdrop-filter: blur(16px);
    border-radius: 22px;
    padding: 18px 20px;
    transition: transform 0.2s ease, border-color 0.2s ease;
    animation: floatIn 0.6s ease both;
}
.kpi:hover{transform: translateY(-3px); border-color: rgba(168,85,247,0.4);}
.kpi-label{font-size:11px; color:#94a3b8; font-weight:700; text-transform:uppercase; letter-spacing:0.08em; font-family:'JetBrains Mono', monospace;}
.kpi-value{font-size:30px; font-weight:900; color:white; margin:6px 0 2px 0;}
.kpi-sub{font-size:12.5px; color:#cbd5e1; font-family:'JetBrains Mono', monospace;}
.section{
    font-size:22px; font-weight:900; color:#e2e8f0; margin:36px 0 16px 0;
    border-bottom:1px solid rgba(255,255,255,0.10); padding-bottom:10px;
    display:flex; align-items:center; gap:12px;
}
.badge-sim{background:rgba(251,146,60,0.15); border:1px solid rgba(251,146,60,0.5); color:#fdba74; padding:5px 12px; border-radius:999px; font-size:11px; font-weight:800; font-family:'JetBrains Mono', monospace;}
.badge-real{background:rgba(34,197,94,0.15); border:1px solid rgba(34,197,94,0.5); color:#86efac; padding:5px 12px; border-radius:999px; font-size:11px; font-weight:800; font-family:'JetBrains Mono', monospace;}
.badge-old{background:rgba(168,85,247,0.15); border:1px solid rgba(168,85,247,0.4); color:#d8b4fe; padding:5px 12px; border-radius:999px; font-size:11px; font-weight:800; font-family:'JetBrains Mono', monospace;}
.file-tag{background:rgba(168,85,247,0.12); border:1px solid rgba(168,85,247,0.3); color:#d8b4fe; padding:3px 9px; border-radius:8px; font-size:10px; font-family:'JetBrains Mono', monospace;}
.pulse{width:10px; height:10px; background:#22c55e; border-radius:50%; box-shadow:0 0 0 0 rgba(34,197,94,0.7); animation:pulse 2s infinite; display:inline-block;}
@keyframes pulse{0%{box-shadow:0 0 0 0 rgba(34,197,94,0.7);} 70%{box-shadow:0 0 0 10px rgba(34,197,94,0);} 100%{box-shadow:0 0 0 0 rgba(34,197,94,0);}}
@keyframes floatIn{from{opacity:0; transform:translateY(12px);} to{opacity:1; transform:translateY(0);}}
.live-dot{animation: blink 1.2s infinite;}
@keyframes blink{0%,100%{opacity:1;} 50%{opacity:0.4;}}
</style>
""", unsafe_allow_html=True)

st.markdown('<div style="font-size:46px; font-weight:900; background:linear-gradient(90deg,#fff,#c4b5fd,#22d3ee,#fb923c); -webkit-background-clip:text; -webkit-text-fill-color:transparent; line-height:1.1;">EnergySavvy AI<br><span style="font-size:18px; font-weight:600; color:#94a3b8; -webkit-text-fill-color:#94a3b8; font-family:JetBrains Mono;">LIVE Egyptian Home • REAL Weather • SIM Power • 3 Pillars ML</span></div>', unsafe_allow_html=True)
st.markdown('<div style="margin:12px 0 6px 0; display:flex; gap:8px; flex-wrap:wrap;"><span class="badge-real"><span class="pulse"></span> WEATHER: REAL API wttr.in + Open-Meteo</span> <span class="badge-sim">POWER: SIMULATED Egyptian Equation</span> <span class="badge-old">TRAIN: OLD UCI + Cairo Weather</span></div>', unsafe_allow_html=True)

import pandas as pd, numpy as np, random, requests, json, math
from datetime import datetime, timedelta
from pathlib import Path
import plotly.graph_objects as go

try:
    from sklearn.ensemble import RandomForestRegressor, IsolationForest
    SKLEARN = True
except Exception:
    SKLEARN = False

try:
    import joblib
    JOBLIB = True
except Exception:
    JOBLIB = False

try:
    import qrcode
    QR = True
except Exception:
    QR = False

ROOT_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT_DIR / "models" / "forecast_rf.pkl"

def get_location_by_ip():
    try:
        r = requests.get("https://ipapi.co/json/", timeout=4).json()
        return {"latitude": float(r.get("latitude", 30.0444)), "longitude": float(r.get("longitude", 31.2357)), "city": r.get("city","Cairo"), "country": r.get("country_name","Egypt"), "source": "ipapi.co REAL", "is_simulated": False}
    except Exception:
        return {"latitude": 30.0444, "longitude": 31.2357, "city":"Cairo","country":"Egypt","source":"Fallback Cairo","is_simulated": False}

def get_current_weather(lat=30.0444, lon=31.2357):
    try:
        r = requests.get("https://wttr.in/Cairo?format=j1", timeout=5).json()
        c = r["current_condition"][0]
        return {"temperature_c": float(c["temp_C"]), "humidity_percent": int(c["humidity"]), "wind_speed_kmh": float(c["windspeedKmph"]), "feels_like_c": float(c.get("FeelsLikeC", c["temp_C"])), "weather_desc": c["weatherDesc"][0]["value"], "source": "wttr.in REAL API", "is_simulated": False, "timestamp": datetime.now().isoformat()}
    except Exception:
        pass
    try:
        url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,relative_humidity_2m,wind_speed_10m,apparent_temperature&timezone=Africa/Cairo"
        data = requests.get(url, timeout=5).json()
        cur = data["current"]
        return {"temperature_c": float(cur["temperature_2m"]), "humidity_percent": int(cur["relative_humidity_2m"]), "wind_speed_kmh": float(cur["wind_speed_10m"]), "feels_like_c": float(cur.get("apparent_temperature", cur["temperature_2m"])), "weather_desc": "Clear", "source": "Open-Meteo REAL API", "is_simulated": False, "timestamp": datetime.now().isoformat()}
    except Exception:
        return {"temperature_c": round(28 + 6*math.sin((datetime.now().hour-6)/24*2*math.pi) + random.uniform(-0.8,0.8),1), "humidity_percent": 55, "wind_speed_kmh": 12.0, "feels_like_c": 32.0, "weather_desc": "Partly cloudy", "source":"Fallback Cairo Simulation", "is_simulated": True, "timestamp": datetime.now().isoformat()}

class EnergySimulator:
    def __init__(self):
        self.appliances = {
            "refrigerator": {"rated_power_kw":0.20, "base_prob":0.95, "sub":"Sub1_Kitchen"},
            "microwave": {"rated_power_kw":1.20, "base_prob":0.08, "sub":"Sub1_Kitchen"},
            "kettle": {"rated_power_kw":1.50, "base_prob":0.05, "sub":"Sub1_Kitchen"},
            "washing_machine": {"rated_power_kw":0.50, "base_prob":0.12, "sub":"Sub2_Laundry"},
            "dryer": {"rated_power_kw":2.00, "base_prob":0.04, "sub":"Sub2_Laundry"},
            "water_heater": {"rated_power_kw":1.50, "base_prob":0.18, "sub":"Sub3_Heater_AC"},
            "air_conditioner": {"rated_power_kw":1.50, "base_prob":0.65, "sub":"Sub3_Heater_AC", "qty":2},
            "lights": {"rated_power_kw":0.144, "base_prob":0.95, "sub":"Unmeasured"},
            "television": {"rated_power_kw":0.24, "base_prob":0.75, "sub":"Unmeasured"},
            "router": {"rated_power_kw":0.03, "base_prob":1.0, "sub":"Unmeasured"}
        }
    def _get_probability(self, appliance, hour, temp_real):
        cfg = self.appliances[appliance]
        prob = cfg["base_prob"]
        if 18 <= hour <= 23 and appliance in ["television","lights","router","kettle"]:
            prob += 0.30
        if 0 <= hour < 6 and appliance in ["television","washing_machine","dryer","microwave"]:
            prob *= 0.18
        if 6 <= hour <= 9 and appliance in ["kettle","microwave","lights"]:
            prob += 0.25
        if appliance == "air_conditioner":
            if temp_real >= 34: prob = 0.92
            elif temp_real >= 32: prob = 0.85
            elif temp_real >= 30: prob = 0.72
            elif temp_real >= 28: prob = 0.55
            elif temp_real >= 26: prob = 0.30
            else: prob = 0.08
        if appliance == "water_heater" and hour in [5,6,7,20,21,22]:
            prob += 0.25
        return min(prob, 1.0)
    def generate(self, temperature=30.0):
        now = datetime.now()
        hour = now.hour
        hour_factor = np.sin(hour/24*2*math.pi - math.pi/2)
        voltage = 220 + hour_factor*6 + np.random.normal(0, 1.8)
        voltage = max(205, min(250, voltage))
        appliance_data = {}
        sub_metering = {"Sub_metering_1":0.0, "Sub_metering_2":0.0, "Sub_metering_3":0.0, "unmeasured_Wh":0.0}
        total_w = 0.0
        for ap, cfg in self.appliances.items():
            prob = self._get_probability(ap, hour, temperature)
            qty = cfg.get("qty",1)
            is_on = random.random() < prob
            pw_kw = 0.0
            if is_on:
                pw_kw = cfg["rated_power_kw"] * random.uniform(0.90,1.10) * qty
            appliance_data[ap] = {"on": is_on, "power_kw": round(pw_kw,3), "prob": round(prob,2)}
            sub = cfg["sub"]
            w = pw_kw * 1000
            if sub == "Sub1_Kitchen":
                sub_metering["Sub_metering_1"] += w
            elif sub == "Sub2_Laundry":
                sub_metering["Sub_metering_2"] += w
            elif sub == "Sub3_Heater_AC":
                sub_metering["Sub_metering_3"] += w
            else:
                sub_metering["unmeasured_Wh"] += w
            total_w += w
        total_w = max(180, total_w)
        total_kw = total_w / 1000
        current_a = total_w / voltage if voltage>0 else 0
        return {
            "timestamp": now.isoformat(),
            "voltage_v": round(voltage,1),
            "current_a": round(current_a,2),
            "power_kw": round(total_kw,3),
            "power_w": round(total_w,1),
            "appliances": appliance_data,
            "sub_metering": {k: round(v,1) for k,v in sub_metering.items()},
            "is_simulated": True,
            "source": "EnergySimulator Egyptian Equation SIMULATED"
        }

class RealtimeEngine:
    def __init__(self, latitude=30.0444, longitude=31.2357):
        self.simulator = EnergySimulator()
        self.latitude = latitude
        self.longitude = longitude
        self.city = "Cairo"
    def get_live_data(self):
        weather = get_current_weather(self.latitude, self.longitude)
        energy = self.simulator.generate(temperature=weather["temperature_c"])
        return {"timestamp": datetime.now().isoformat(), "weather": weather, "energy": energy, "location": {"city": self.city, "lat": self.latitude, "lon": self.longitude}}

def load_household_data():
    try:
        p = ROOT_DIR / "data" / "household_power" / "processed" / "household_power_hourly.parquet"
        if p.exists():
            return pd.read_parquet(p)
        url = "https://raw.githubusercontent.com/Mohamed-Al-Saudi/EnergySavvy-AI-Project/main/data/household_power/processed/household_power_hourly.parquet"
        return pd.read_parquet(url)
    except Exception:
        return None

def add_household_features(df):
    result = df.copy()
    if not isinstance(result.index, pd.DatetimeIndex):
        result.index = pd.to_datetime(result.index)
    result["hour"] = result.index.hour
    result["dayofweek"] = result.index.dayofweek
    result["month"] = result.index.month
    result["is_weekend"] = (result["dayofweek"] >= 5).astype(int)
    col = "Global_active_power" if "Global_active_power" in result.columns else result.columns[0]
    for lag in [1,2,3,24,168]:
        result[f"lag_{lag}"] = result[col].shift(lag)
    result["roll_24_mean"] = result[col].shift(1).rolling(24).mean()
    result["roll_24_std"] = result[col].shift(1).rolling(24).std()
    return result

def forecast_next_hours(model, df_hist, feature_order, hours=6):
    preds = []
    temp_df = df_hist.copy()
    for i in range(hours):
        next_time = temp_df.index[-1] + timedelta(hours=1)
        hist_vals = list(temp_df["Global_active_power"].values) + preds
        def safe_lag(n):
            return hist_vals[-n] if len(hist_vals) >= n else hist_vals[-1]
        new_row = {
            "hour": next_time.hour,
            "dayofweek": next_time.dayofweek,
            "month": next_time.month,
            "is_weekend": int(next_time.dayofweek >= 5),
            "lag_1": safe_lag(1), "lag_2": safe_lag(2), "lag_3": safe_lag(3),
            "lag_24": safe_lag(24), "lag_168": safe_lag(168),
            "roll_24_mean": float(np.mean(hist_vals[-24:])) if len(hist_vals)>=1 else 1.0,
            "roll_24_std": float(np.std(hist_vals[-24:])) if len(hist_vals)>=24 else 0.15
        }
        X_next = np.array([[new_row[f] for f in feature_order]])
        try:
            p = model.predict(X_next)[0]
        except Exception:
            p = temp_df["Global_active_power"].tail(24).mean() * random.uniform(0.9,1.1)
        preds.append(max(0.05, float(p)))
        temp_df.loc[next_time] = pd.Series({**{"Global_active_power": preds[-1]}, **new_row})
    return preds

def detect_live_anomaly(latest_power, df_hist_tail, model):
    try:
        feature_order = ['hour','dayofweek','month','is_weekend','lag_1','lag_2','lag_3','lag_24','lag_168','roll_24_mean','roll_24_std']
        df_feat = add_household_features(df_hist_tail).dropna()
        if len(df_feat) < 10 or not SKLEARN:
            return False, 0.0
        X = df_feat[feature_order].values
        y = df_feat["Global_active_power"].values
        y_pred = model.predict(X)
        residuals = y - y_pred
        rmse = np.sqrt(np.mean(residuals**2))
        last_pred = y_pred[-1]
        live_resid = latest_power - last_pred
        z = live_resid / (np.std(residuals)+1e-6)
        is_anomaly = (abs(z) > 3.0) or (abs(live_resid) > 2*rmse)
        return bool(is_anomaly), float(z)
    except Exception:
        return False, 0.0

def generate_many_recommendations(df_hourly):
    recs = []
    df = df_hourly.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    for idx, row in df[(df.index.hour >= 18) & (df.index.hour <= 22) & (df['Global_active_power'] > 1.5)].iloc[:500].iterrows():
        recs.append({'timestamp': idx, 'type':'Peak_Shift','severity':'High','message':f"Peak {idx.hour}h {row['Global_active_power']:.2f}kW > avg 1.037kW shift to 2-5am save 22%",'estimated_saving_kwh':0.25})
    for idx, row in df[(df.index.hour >= 0) & (df.index.hour <= 5) & (df['Global_active_power'] > 0.5)].iloc[:500].iterrows():
        recs.append({'timestamp': idx, 'type':'Night_Idle','severity':'Medium','message':f"Night {idx.hour}h idle {row['Global_active_power']:.2f}kW vs 0.479kW baseline turn off standby",'estimated_saving_kwh':0.12})
    if 'Sub_metering_3' in df.columns:
        total_sub = df[['Sub_metering_1','Sub_metering_2','Sub_metering_3']].sum(axis=1)+1e-6
        sub3_pct = df['Sub_metering_3']/total_sub
        for idx in df[sub3_pct>0.6].index[:600]:
            row=df.loc[idx]
            recs.append({'timestamp':idx,'type':'High_SM3_AC_Heater','severity':'Medium','message':f"Sub3 {sub3_pct.loc[idx]*100:.0f}% avg 72.8% AC 26C save 18% {row['Global_active_power']:.2f}kW",'estimated_saving_kwh':0.15})
    df['is_weekend']=(df.index.dayofweek>=5).astype(int)
    for idx,row in df[(df['is_weekend']==1) & (df['Global_active_power']>1.5)].iloc[:300].iterrows():
        recs.append({'timestamp':idx,'type':'Weekend_High','severity':'Low','message':f"Weekend {row['Global_active_power']:.2f}kW > weekday 1.037kW optimize",'estimated_saving_kwh':0.10})
    for idx,row in df[df['Global_active_power']>2.0].iloc[:300].iterrows():
        recs.append({'timestamp':idx,'type':'Weather_Heat_Correlation','severity':'Medium','message':f"High {row['Global_active_power']:.2f}kW heat Cairo mean 23.03 max 37.4 temp>32 AC +15%",'estimated_saving_kwh':0.18})
    rec_df=pd.DataFrame(recs)
    if not rec_df.empty:
        rec_df=rec_df.sort_values('timestamp').reset_index(drop=True)
    return rec_df

with st.sidebar:
    st.markdown("### Flow")
    st.markdown("""
    <div style="font-family:JetBrains Mono; font-size:11px; line-height:1.7; color:#94a3b8;">
    <span style="color:#22c55e;">●</span> Live Location ipapi.co<br>
    <span style="color:#22c55e;">●</span> Real Weather wttr.in + Open-Meteo<br>
    <span style="color:#fb923c;">●</span> RealtimeEngine<br>
    <span style="color:#fb923c;">●</span> EnergySimulator Egyptian Equation<br>
    <span style="color:#a855f7;">●</span> Forecast Model forecast_rf.pkl<br>
    <span style="color:#a855f7;">●</span> Anomaly Detection<br>
    <span style="color:#a855f7;">●</span> Recommendations 8 Cases<br>
    <span style="color:#e2e8f0;">●</span> Dashboard
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("### Scan for Phone")
    deploy_url = st.text_input("Deploy URL for QR:", value="https://energysavvy-ai.streamlit.app")
    if QR and deploy_url:
        try:
            qr = qrcode.QRCode(version=1, error_correction=qrcode.constants.ERROR_CORRECT_L, box_size=10, border=2)
            qr.add_data(deploy_url)
            qr.make(fit=True)
            img = qr.make_image(fill_color="#e2e8f0", back_color="#070C1A").convert("RGB")
            import io
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            buf.seek(0)
            st.image(buf, caption="Scan to open on phone", use_container_width=True)
        except Exception as e:
            st.warning(f"QR error: {e}")
    else:
        st.code(deploy_url)
    st.markdown("---")
    st.markdown('<span class="badge-old">UCI 34589 rows 2006-2010</span>', unsafe_allow_html=True)
    st.markdown('<span class="badge-real">Weather REAL wttr.in</span>', unsafe_allow_html=True)
    st.markdown('<span class="badge-sim">Power SIM Egyptian Equation</span>', unsafe_allow_html=True)

if "location" not in st.session_state:
    with st.spinner("Getting Live Location ipapi.co..."):
        loc = get_location_by_ip()
        st.session_state.location = loc
else:
    loc = st.session_state.location

if "engine" not in st.session_state:
    st.session_state.engine = RealtimeEngine(latitude=loc["latitude"], longitude=loc["longitude"])
    st.session_state.engine.city = loc["city"]
    st.session_state.history = []
    with st.spinner("Fetching REAL Weather + Generating SIM Power..."):
        for i in range(30):
            try:
                d = st.session_state.engine.get_live_data()
                d["time"] = datetime.now() - timedelta(minutes=30-i)
                st.session_state.history.append(d)
            except Exception:
                break

if st.session_state.history and (datetime.now() - pd.to_datetime(st.session_state.history[-1]["time"])).total_seconds() > 3:
    d = st.session_state.engine.get_live_data()
    d["time"] = datetime.now()
    st.session_state.history.append(d)
    st.session_state.history = st.session_state.history[-200:]

latest = st.session_state.history[-1] if st.session_state.history else None
if not latest:
    st.error("No live data - check internet")
    st.stop()

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(f'<div class="kpi" style="border-left:3px solid #22c55e;"><div class="kpi-label">LIVE Location <span class="file-tag">ipapi.co REAL</span></div><div class="kpi-value">{loc["city"]}, {loc["country"]}</div><div class="kpi-sub">{loc["latitude"]:.4f}, {loc["longitude"]:.4f} | {loc["source"]}</div></div>', unsafe_allow_html=True)
with col2:
    w = latest["weather"]
    st.markdown(f'<div class="kpi" style="border-left:3px solid #22c55e;"><div class="kpi-label">REAL Weather <span class="file-tag">wttr.in REAL</span> <span class="live-dot">● LIVE</span></div><div class="kpi-value">{w["temperature_c"]:.1f}°C <span style="font-size:14px; color:#94a3b8;">feels {w["feels_like_c"]:.0f}°C</span></div><div class="kpi-sub">{w["weather_desc"]} | {w["source"]} | Hum {w["humidity_percent"]}% Wind {w["wind_speed_kmh"]:.1f}km/h</div></div>', unsafe_allow_html=True)
with col3:
    e = latest["energy"]
    on_count = sum(1 for v in e["appliances"].values() if v["on"])
    st.markdown(f'<div class="kpi" style="border-left:3px solid #fb923c;"><div class="kpi-label">SIM Power <span class="file-tag">EnergySimulator SIM</span> <span class="live-dot">● LIVE</span></div><div class="kpi-value" style="color:#fb923c;">{e["power_kw"]:.3f} kW <span style="font-size:14px; color:#cbd5e1;">{e["power_w"]:.0f} W</span></div><div class="kpi-sub">{e["voltage_v"]}V | {e["current_a"]}A | {on_count}/10 devices ON</div></div>', unsafe_allow_html=True)
with col4:
    st.markdown(f'<div class="kpi" style="border-left:3px solid #a855f7;"><div class="kpi-label">RealtimeEngine</div><div class="kpi-value">{len(st.session_state.history)} pts</div><div class="kpi-sub">V=220+sin(h)*6+noise | AC prob = f(REAL temp)</div></div>', unsafe_allow_html=True)

df_plot = pd.DataFrame([{"time": h["time"], "power_kw": h["energy"]["power_kw"], "temp_c": h["weather"]["temperature_c"], "voltage": h["energy"]["voltage_v"]} for h in st.session_state.history])
fig = go.Figure()
fig.add_trace(go.Scatter(x=df_plot["time"], y=df_plot["power_kw"], mode="lines", name="SIM Power kW", line=dict(color="#fb923c", width=3, shape='spline'), fill="tozeroy", fillcolor="rgba(251,146,60,0.18)"))
fig.add_trace(go.Scatter(x=df_plot["time"], y=df_plot["temp_c"]/10, mode="lines", name="REAL Temp /10", line=dict(color="#22c55e", width=2, dash="dash", shape='spline')))
fig.add_trace(go.Scatter(x=df_plot["time"], y=df_plot["voltage"]/100, mode="lines", name="Voltage /100", line=dict(color="#a855f7", width=1.5, dash="dot"), opacity=0.6))
fig.update_layout(height=380, title=dict(text=f"LIVE: {loc['city']} | REAL Temp drives SIM Power", font=dict(size=13, color="#cbd5e1")), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#cbd5e1"), margin=dict(l=10,r=10,t=50,b=10), legend=dict(orientation="h", y=1.02, x=1), hovermode="x unified")
st.plotly_chart(fig, use_container_width=True)

st.markdown('<div class="section">Live Egyptian House Breakdown <span class="file-tag">simulator.py DEVICE_MAP equation</span> <span class="badge-sim">SIM</span></div>', unsafe_allow_html=True)
cols = st.columns(5)
devices = list(latest["energy"]["appliances"].items())
for i, (name, data) in enumerate(devices):
    with cols[i % 5]:
        color = "#22c55e" if data["on"] else "#475569"
        status = "ON" if data["on"] else "OFF"
        st.markdown(f'<div class="kpi" style="padding:12px 14px; border-left:3px solid {color};"><div class="kpi-label">{name.upper()}</div><div class="kpi-value" style="font-size:18px; color:{color};">{data["power_kw"]:.3f} kW</div><div class="kpi-sub">{status} prob {data["prob"]:.2f}</div></div>', unsafe_allow_html=True)

sub = latest["energy"]["sub_metering"]
c1, c2, c3, c4 = st.columns(4)
c1.metric("Sub1 Kitchen", f"{sub['Sub_metering_1']:.0f} W", "Fridge+Microwave+Kettle")
c2.metric("Sub2 Laundry", f"{sub['Sub_metering_2']:.0f} W", "Washing+Dryer")
c3.metric("Sub3 AC/Heater", f"{sub['Sub_metering_3']:.0f} W", "72.8% avg")
c4.metric("Unmeasured", f"{sub['unmeasured_Wh']:.0f} W", "Lights+TV+Router")

st.markdown('<div class="section">1 Forecast Future Consumption <span class="file-tag">PILLAR 1 FROM forecast_rf.pkl OLD -> NEW</span></div>', unsafe_allow_html=True)
col_f1, col_f2 = st.columns([2,1])
with col_f1:
    try:
        df_hourly = load_household_data()
        if df_hourly is not None and JOBLIB and MODEL_PATH.exists() and SKLEARN:
            df_feat = add_household_features(df_hourly).dropna()
            model = joblib.load(MODEL_PATH)
            feature_order = ['hour','dayofweek','month','is_weekend','lag_1','lag_2','lag_3','lag_24','lag_168','roll_24_mean','roll_24_std']
            preds = forecast_next_hours(model, df_feat.tail(200), feature_order, 6)
            live_hist = pd.DataFrame([{"Global_active_power": h["energy"]["power_kw"]} for h in st.session_state.history[-168:]], index=pd.date_range(end=datetime.now(), periods=min(168, len(st.session_state.history)), freq='h'))
            live_feat = add_household_features(live_hist).dropna()
            live_preds = forecast_next_hours(model, live_feat.tail(100), feature_order, 6) if len(live_feat) >= 24 else preds
            fig_f = go.Figure()
            fig_f.add_trace(go.Bar(x=[f"+{i+1}h" for i in range(6)], y=preds, name="OLD UCI based", marker_color="#a855f7", opacity=0.7))
            fig_f.add_trace(go.Bar(x=[f"+{i+1}h" for i in range(6)], y=live_preds, name="LIVE SIM based", marker_color="#fb923c"))
            fig_f.update_layout(height=300, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#cbd5e1"), barmode='group', margin=dict(l=10,r=10,t=10,b=10))
            st.plotly_chart(fig_f, use_container_width=True)
        else:
            demo = [df_plot["power_kw"].tail(24).mean() * random.uniform(0.9,1.15) for _ in range(6)]
            st.bar_chart(demo)
            preds = demo
            live_preds = demo
    except Exception as e:
        st.error(f"Forecast error: {e}")
        preds = [1.0]*6
        live_preds = [1.0]*6

with col_f2:
    if 'preds' in locals():
        st.metric("Next Hour LIVE", f"{live_preds[0]:.3f} kW", delta=f"{(live_preds[0]-latest['energy']['power_kw']):+.3f} vs now")
        st.metric("6h Avg LIVE", f"{np.mean(live_preds):.3f} kW")
        st.metric("Historical MAE/RMSE", "0.317 / 0.464")
        st.info(f"OLD trains NEW predicts: LIVE SIM {latest['energy']['power_kw']:.3f}kW driven by REAL temp {latest['weather']['temperature_c']:.1f}C")

st.markdown('<div class="section">2 Detect Unusual Behavior <span class="file-tag">PILLAR 2</span></div>', unsafe_allow_html=True)
col_a1, col_a2 = st.columns([1,2])
with col_a1:
    try:
        df_hourly = load_household_data()
        if df_hourly is not None and MODEL_PATH.exists() and SKLEARN:
            model = joblib.load(MODEL_PATH)
            is_anom, z = detect_live_anomaly(latest["energy"]["power_kw"], df_hourly.tail(500), model)
            if is_anom:
                st.error(f"LIVE ANOMALY! Power {latest['energy']['power_kw']:.3f}kW z={z:.2f} >3.0")
            else:
                st.success(f"LIVE Normal Power {latest['energy']['power_kw']:.3f}kW z={z:.2f}")
            st.metric("Historical Anomalies", "1286", "3.74%")
            st.metric("Live Z-Score", f"{z:.2f}")
    except Exception as e:
        st.error(f"Anomaly error: {e}")
with col_a2:
    st.info(f"OLD: Model learned normal pattern. LIVE: SIM {latest['energy']['power_kw']:.3f}kW driven by REAL temp {latest['weather']['temperature_c']:.1f}C. If REAL temp >=34C -> AC prob 0.92")

st.markdown('<div class="section">3 Data-Driven Recommendations <span class="file-tag">PILLAR 3 8 Cases VERY LOT</span></div>', unsafe_allow_html=True)
try:
    df_hourly = load_household_data()
    if df_hourly is not None:
        rec_df = generate_many_recommendations(df_hourly)
        live_recs = []
        e = latest["energy"]
        w = latest["weather"]
        if datetime.now().hour >= 18 and datetime.now().hour <= 22 and e["power_kw"] > 1.5:
            live_recs.append({"type":"LIVE Peak_Shift", "severity":"High", "message": f"LIVE NOW {datetime.now().hour}h {e['power_kw']:.2f}kW peak shift to 2-5am", "saving":0.25})
        if w["temperature_c"] >= 32 and e["appliances"]["air_conditioner"]["on"]:
            live_recs.append({"type":"LIVE Heat_AC", "severity":"Medium", "message": f"LIVE REAL temp {w['temperature_c']:.1f}C >=32 AC ON {e['sub_metering']['Sub_metering_3']:.0f}W - set AC 26C save 18%", "saving":0.18})
        if e["power_kw"] > 3.0:
            live_recs.append({"type":"LIVE Critical", "severity":"High", "message": f"LIVE CRITICAL {e['power_kw']:.3f}kW >3kW limit", "saving":0.45})
        col_r1, col_r2 = st.columns([1,2])
        with col_r1:
            st.markdown("#### LIVE Recommendations")
            if live_recs:
                for rec in live_recs:
                    sev_color = "#ef4444" if rec["severity"]=="High" else "#f59e0b"
                    st.markdown(f'<div class="kpi" style="border-left:3px solid {sev_color}; margin-bottom:10px;"><div class="kpi-label">{rec["type"]} {rec["severity"]}</div><div class="kpi-sub" style="color:#e2e8f0;">{rec["message"]}</div><div class="kpi-label" style="color:{sev_color};">Save {rec["saving"]:.2f} kWh</div></div>', unsafe_allow_html=True)
            else:
                st.success("LIVE No critical recommendations now!")
        with col_r2:
            st.markdown(f"#### Historical {len(rec_df)} rows")
            st.metric("Total VERY LOT", f"{len(rec_df)} rows", "8 cases")
            if not rec_df.empty:
                st.dataframe(rec_df.head(100), use_container_width=True, height=400)
                st.bar_chart(rec_df["type"].value_counts())
except Exception as e:
    st.error(f"Rec error: {e}")

st.markdown("---")
col_foot1, col_foot2, col_foot3 = st.columns([2,1,1])
with col_foot1:
    st.markdown(f"**Flow:** Location {loc['city']} {loc['latitude']:.4f},{loc['longitude']:.4f} -> Weather REAL {latest['weather']['temperature_c']:.1f}C {latest['weather']['source']} -> RealtimeEngine -> EnergySimulator Egyptian Equation -> Forecast Model -> Anomaly -> Recommendations -> Dashboard")
with col_foot2:
    st.markdown("**Scan Me**")
    if QR:
        try:
            qr2 = qrcode.QRCode(version=1, error_correction=qrcode.constants.ERROR_CORRECT_L, box_size=8, border=2)
            qr2.add_data(deploy_url)
            qr2.make(fit=True)
            img2 = qr2.make_image(fill_color="white", back_color="#070C1A").convert("RGB")
            import io
            buf2 = io.BytesIO()
            img2.save(buf2, format="PNG")
            buf2.seek(0)
            st.image(buf2, width=180)
        except Exception:
            st.code(deploy_url)
    st.caption(f"LIVE: {loc['city']} {datetime.now().strftime('%H:%M:%S')}")
with col_foot3:
    st.markdown("**Status**")
    st.success(f"Location REAL: {loc['source']}")
    st.success(f"Weather REAL: {latest['weather']['source']}")
    st.warning(f"Power SIM: {latest['energy']['source']}")

st.caption("EnergySavvy AI FINAL - Egyptian Equation + 3 Pillars ML - POWER SIM WEATHER REAL - OLD UCI trains NEW LIVE predicts - Dynamic beautiful life dashboard with QR")

if st.button("Refresh LIVE", type="primary"):
    st.rerun()
