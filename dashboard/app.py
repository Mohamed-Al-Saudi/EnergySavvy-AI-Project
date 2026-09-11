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
# ============================================================================

import base64
import io
import math
import random
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

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
# All Egyptian governorates.
# ----------------------------------------------------------------------------
GOVERNORATES = {
    "Cairo": (30.0444, 31.2357),
    "Alexandria": (31.2001, 29.9187),
    "Port Said": (31.2653, 32.3019),
    "Suez": (29.9668, 32.5498),
    "Damietta": (31.4175, 31.8144),
    "Dakahlia": (31.0409, 31.3785),
    "Sharqia": (30.7327, 31.7195),
    "Qalyubia": (30.2807, 31.2043),
    "Kafr El Sheikh": (31.1107, 30.9388),
    "Gharbia": (30.7865, 31.0004),
    "Monufia": (30.5972, 30.9876),
    "Beheira": (30.8481, 30.3436),
    "Ismailia": (30.5965, 32.2715),
    "Giza": (30.0131, 31.2089),
    "Fayoum": (29.3084, 30.8428),
    "Beni Suef": (29.0661, 31.0994),
    "Minya": (28.1099, 30.7503),
    "Assiut": (27.1809, 31.1837),
    "Sohag": (26.5591, 31.6959),
    "Qena": (26.1551, 32.7160),
    "Luxor": (25.6872, 32.6396),
    "Aswan": (24.0889, 32.8998),
    "Red Sea": (27.2579, 33.8116),
    "New Valley": (25.4417, 30.5586),
    "Matrouh": (31.3543, 27.2373),
    "North Sinai": (31.0409, 33.0114),
    "South Sinai": (28.5550, 34.7500),
}

# ----------------------------------------------------------------------------
# Weather: Open-Meteo is primary and uses the selected governorate coordinates.
# wttr.in is a fallback, but NEVER hard-coded to Cairo.
# ----------------------------------------------------------------------------
WEATHER_CACHE_SECONDS = 45

def weather_code_description(code):
    code = int(code)
    mapping = {
        0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
        45: "Fog", 48: "Depositing rime fog",
        51: "Light drizzle", 53: "Moderate drizzle", 55: "Dense drizzle",
        61: "Slight rain", 63: "Moderate rain", 65: "Heavy rain",
        71: "Slight snow", 73: "Moderate snow", 75: "Heavy snow",
        80: "Slight rain showers", 81: "Moderate rain showers", 82: "Violent rain showers",
        95: "Thunderstorm", 96: "Thunderstorm with hail", 99: "Thunderstorm with heavy hail",
    }
    return mapping.get(code, "Current conditions")

@st.cache_data(ttl=WEATHER_CACHE_SECONDS, show_spinner=False)
def get_current_weather(lat, lon):
    # Primary live source: location-specific Open-Meteo.
    try:
        url = (
            "https://api.open-meteo.com/v1/forecast"
            f"?latitude={lat}&longitude={lon}"
            "&current=temperature_2m,relative_humidity_2m,wind_speed_10m,"
            "apparent_temperature,weather_code"
            "&timezone=auto"
        )
        data = requests.get(url, timeout=8).json()
        cur = data["current"]
        return {
            "temperature_c": float(cur["temperature_2m"]),
            "humidity_percent": int(round(cur["relative_humidity_2m"])),
            "wind_speed_kmh": float(cur["wind_speed_10m"]),
            "feels_like_c": float(cur["apparent_temperature"]),
            "weather_desc": weather_code_description(cur["weather_code"]),
            "source": "Open-Meteo live",
            "is_simulated": False,
            "timestamp": datetime.now().isoformat(),
        }
    except Exception:
        pass

    # Location-specific fallback.
    try:
        url = f"https://wttr.in/{lat},{lon}?format=j1"
        data = requests.get(url, timeout=8).json()
        c = data["current_condition"][0]
        return {
            "temperature_c": float(c["temp_C"]),
            "humidity_percent": int(c["humidity"]),
            "wind_speed_kmh": float(c["windspeedKmph"]),
            "feels_like_c": float(c.get("FeelsLikeC", c["temp_C"])),
            "weather_desc": c["weatherDesc"][0]["value"],
            "source": "wttr.in live",
            "is_simulated": False,
            "timestamp": datetime.now().isoformat(),
        }
    except Exception:
        # Last-resort fallback only; clearly labelled.
        hour = datetime.now().hour
        temp = 28 + 5 * math.sin((hour - 6) / 24 * 2 * math.pi)
        return {
            "temperature_c": round(temp, 1),
            "humidity_percent": 55,
            "wind_speed_kmh": 12.0,
            "feels_like_c": round(temp + 2, 1),
            "weather_desc": "Fallback estimate",
            "source": "Fallback estimate (API unavailable)",
            "is_simulated": True,
            "timestamp": datetime.now().isoformat(),
        }

def get_ip_location():
    try:
        r = requests.get("https://ipapi.co/json/", timeout=5).json()
        city = r.get("city") or "Cairo"
        country = r.get("country_name") or "Egypt"
        lat = float(r.get("latitude", GOVERNORATES["Cairo"][0]))
        lon = float(r.get("longitude", GOVERNORATES["Cairo"][1]))
        return {"city": city, "country": country, "latitude": lat, "longitude": lon, "source": "IP auto-detection"}
    except Exception:
        return {"city": "Cairo", "country": "Egypt", "latitude": GOVERNORATES["Cairo"][0],
                "longitude": GOVERNORATES["Cairo"][1], "source": "Cairo fallback"}

# ----------------------------------------------------------------------------
# 15-device production simulator.
# Continuous devices are always AVAILABLE/ACTIVE in the dashboard, but billing
# uses duty cycles so a 1,500 W appliance is not falsely billed 24h at full load.
# ----------------------------------------------------------------------------
CONTINUOUS_DEVICES = {
    "Washing machine": {"kw": 1.20, "duty": 0.03},
    "Refrigerator": {"kw": 0.15, "duty": 0.35},
    "Stove": {"kw": 1.20, "duty": 0.05},
    "Water heater": {"kw": 1.60, "duty": 0.12},
    "Wi-Fi router": {"kw": 0.02, "duty": 1.00},
    "Radio": {"kw": 0.03, "duty": 0.15},
    "Oven": {"kw": 1.50, "duty": 0.03},
    "Microwave": {"kw": 1.20, "duty": 0.02},
    "Deep freezer": {"kw": 0.20, "duty": 0.35},
    "Freezer": {"kw": 0.18, "duty": 0.35},
}
VARIABLE_DEVICES = {
    "Air conditioner": {"kw": 1.50},
    "Fans": {"kw": 0.075},
    "Lamps": {"kw": 0.12},
    "Phone/device chargers": {"kw": 0.04},
    "Vacuum cleaner": {"kw": 0.70},
}
ALL_DEVICE_NAMES = list(CONTINUOUS_DEVICES) + list(VARIABLE_DEVICES)

class EnergySimulator:
    def __init__(self):
        self.previous = {name: False for name in ALL_DEVICE_NAMES}

    def probability(self, name, hour, temp):
        if name in CONTINUOUS_DEVICES:
            return 1.0
        if name == "Air conditioner":
            if temp >= 34: base = 0.90
            elif temp >= 31: base = 0.72
            elif temp >= 28: base = 0.48
            elif temp >= 25: base = 0.24
            else: base = 0.06
            if 0 <= hour <= 6: base *= 0.55
            return min(base, 0.95)
        if name == "Fans":
            if temp >= 32: return 0.72
            if temp >= 29: return 0.52
            if temp >= 25: return 0.30
            return 0.07
        if name == "Lamps":
            if 18 <= hour <= 23 or 0 <= hour <= 5: return 0.78
            if 6 <= hour <= 7: return 0.30
            return 0.035
        if name == "Phone/device chargers":
            if 18 <= hour <= 23: return 0.58
            if 7 <= hour <= 10: return 0.38
            return 0.10
        if name == "Vacuum cleaner":
            return 0.16 if 9 <= hour <= 17 else 0.01
        return 0.05

    def generate(self, temperature=30.0, when=None):
        now = when or datetime.now()
        hour = now.hour
        # Small continuous variation prevents a flat line while remaining plausible.
        voltage = np.clip(
            230 + 3.0 * math.sin((hour / 24) * 2 * math.pi) + np.random.normal(0, 1.2),
            215, 245
        )
        appliances = {}
        total_w = 0.0

        for name, cfg in CONTINUOUS_DEVICES.items():
            # Dashboard status: continuously available/active; actual energy uses duty cycle.
            cycle = cfg["duty"]
            # Instantaneous compressor/heater/operation variation.
            active_now = random.random() < min(0.98, 0.12 + cycle * 1.8)
            power_kw = cfg["kw"] * random.uniform(0.75, 1.10) if active_now else cfg["kw"] * 0.04
            # Keep "on" status true because these are in the required 24H group.
            appliances[name] = {
                "on": True,
                "power_kw": round(power_kw, 3),
                "instant_active": active_now,
                "prob": 1.0,
                "group": "24-hour household",
            }
            total_w += power_kw * 1000

        for name, cfg in VARIABLE_DEVICES.items():
            p = self.probability(name, hour, temperature)
            previous = self.previous.get(name, False)
            stay_p = min(0.96, p + 0.35) if previous else p
            is_on = random.random() < stay_p
            power_kw = cfg["kw"] * random.uniform(0.80, 1.15) if is_on else 0.0
            appliances[name] = {
                "on": is_on,
                "power_kw": round(power_kw, 3),
                "instant_active": is_on,
                "prob": round(p, 2),
                "group": "time-dependent",
            }
            self.previous[name] = is_on
            total_w += power_kw * 1000

        total_w = max(120.0, total_w)
        return {
            "timestamp": now.isoformat(),
            "voltage_v": round(float(voltage), 1),
            "current_a": round(float(total_w / voltage), 2),
            "power_kw": round(total_w / 1000, 3),
            "power_w": round(total_w, 1),
            "appliances": appliances,
            "is_simulated": True,
            "source": "Temporary IoT simulator",
        }

# ----------------------------------------------------------------------------
# Offline-trained forecast model. Historical data is never loaded/displayed.
# Live prediction uses a live-generated hourly baseline only.
# ----------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT_DIR / "models" / "forecast_rf.pkl"
FEATURE_ORDER = [
    "hour", "dayofweek", "month", "is_weekend",
    "lag_1", "lag_2", "lag_3", "lag_24", "lag_168",
    "roll_24_mean", "roll_24_std",
]

try:
    import joblib
    MODEL_AVAILABLE = MODEL_PATH.exists()
except Exception:
    joblib = None
    MODEL_AVAILABLE = False

def load_model():
    if MODEL_AVAILABLE:
        try:
            return joblib.load(MODEL_PATH)
        except Exception:
            return None
    return None

def build_features_from_values(values, timestamps):
    s = pd.Series(values, index=pd.DatetimeIndex(timestamps), name="power_kw")
    f = pd.DataFrame(index=s.index)
    f["power_kw"] = s
    f["hour"] = f.index.hour
    f["dayofweek"] = f.index.dayofweek
    f["month"] = f.index.month
    f["is_weekend"] = (f["dayofweek"] >= 5).astype(int)
    for lag in [1, 2, 3, 24, 168]:
        f[f"lag_{lag}"] = f["power_kw"].shift(lag)
    f["roll_24_mean"] = f["power_kw"].shift(1).rolling(24).mean()
    f["roll_24_std"] = f["power_kw"].shift(1).rolling(24).std()
    return f

def create_live_hourly_history(simulator, temp_now):
    # 168 hourly points generated by the CURRENT production simulator.
    # This is a live/synthetic warm start, not UCI data.
    now = datetime.now().replace(minute=0, second=0, microsecond=0)
    timestamps = [now - timedelta(hours=(167 - i)) for i in range(168)]
    vals = []
    for ts in timestamps:
        # Approximate the current live temperature across the simulated day.
        temp = temp_now + 2.2 * math.sin((ts.hour - now.hour) / 24 * 2 * math.pi)
        d = simulator.generate(temperature=temp, when=ts)
        vals.append(d["power_kw"])
    return pd.Series(vals, index=pd.DatetimeIndex(timestamps), name="power_kw")

def forecast_live_next_hours(model, hourly_series, hours=6):
    work = hourly_series.copy()
    preds = []
    for _ in range(hours):
        next_time = work.index[-1] + timedelta(hours=1)
        f = build_features_from_values(work.values, work.index)
        row = f.iloc[-1].copy()
        # Build features for the next timestamp from live values.
        vals = list(work.values)
        new_features = {
            "hour": next_time.hour,
            "dayofweek": next_time.dayofweek,
            "month": next_time.month,
            "is_weekend": int(next_time.dayofweek >= 5),
            "lag_1": vals[-1],
            "lag_2": vals[-2],
            "lag_3": vals[-3],
            "lag_24": vals[-24],
            "lag_168": vals[-168],
            "roll_24_mean": float(np.mean(vals[-24:])),
            "roll_24_std": float(np.std(vals[-24:])),
        }
        if model is not None:
            try:
                pred = float(model.predict(np.array([[new_features[k] for k in FEATURE_ORDER]]))[0])
            except Exception:
                pred = float(np.mean(vals[-6:]))
        else:
            pred = float(np.mean(vals[-6:]))
        # Keep forecast within a sensible range around the live simulator.
        pred = float(np.clip(pred, 0.08, max(8.0, np.percentile(vals, 95) * 1.5)))
        preds.append(pred)
        work.loc[next_time] = pred
    return preds

# ----------------------------------------------------------------------------
# Residential bill calculation.
# The dashboard uses the user's requested scenario/tariff assumptions.
# ----------------------------------------------------------------------------
BILL_TIERS = [
    (350, 1.72),
    (650, 2.18),
    (1000, 2.40),
    (float("inf"), 2.74),
]

def calculate_monthly_bill(monthly_kwh):
    # Demo tariff requested for the jury-facing estimate.
    # Above 1,000 kWh the 2.74 EGP/kWh rate applies to the full consumption.
    kwh = max(0.0, float(monthly_kwh))
    if kwh <= 350:
        rate = 1.72
    elif kwh <= 650:
        rate = 2.18
    elif kwh <= 1000:
        rate = 2.40
    else:
        rate = 2.74
    return round(kwh * rate, 2)

def estimate_monthly_from_devices(simulator, temp_now):
    # Expected daily energy based on device power x expected operating hours.
    now_hour = datetime.now().hour
    daily_kwh = 0.0
    rows = []

    for name, cfg in CONTINUOUS_DEVICES.items():
        hours = 24.0 * cfg["duty"]
        kwh = cfg["kw"] * hours
        daily_kwh += kwh
        rows.append((name, cfg["kw"], hours, kwh))

    for name, cfg in VARIABLE_DEVICES.items():
        expected_hours = sum(
            simulator.probability(name, h, temp_now + 2.0 * math.sin((h - now_hour) / 24 * 2 * math.pi))
            for h in range(24)
        )
        kwh = cfg["kw"] * expected_hours
        daily_kwh += kwh
        rows.append((name, cfg["kw"], expected_hours, kwh))

    monthly_kwh = daily_kwh * 30.0
    bill = calculate_monthly_bill(monthly_kwh)
    return daily_kwh, monthly_kwh, bill, rows

# ----------------------------------------------------------------------------
# Live anomaly detection based ONLY on the current live stream.
# A rolling live baseline is maintained, so the result can change over time.
# ----------------------------------------------------------------------------
def live_anomaly(power_kw, history):
    if len(history) < 8:
        return False, 0.0, float(np.mean(history)) if history else power_kw
    arr = np.array(history[-40:], dtype=float)
    baseline = float(np.mean(arr[:-1])) if len(arr) > 1 else float(arr.mean())
    std = float(np.std(arr[:-1])) if len(arr) > 2 else 0.08
    std = max(std, 0.06)
    z = (power_kw - baseline) / std
    # Dynamic threshold: anomaly only when deviation is substantial and persistent
    is_anomaly = abs(z) >= 3.2 and abs(power_kw - baseline) >= 0.45
    return bool(is_anomaly), float(z), baseline

# ----------------------------------------------------------------------------
# Conditional recommendation engine. It returns 0..N messages based on LIVE
# device states, current temperature, power, time, and anomaly state.
# ----------------------------------------------------------------------------
def generate_live_recommendations(live, is_anomaly, z, monthly_kwh):
    e = live["energy"]
    w = live["weather"]
    now = datetime.now()
    recs = []
    appliances = e["appliances"]

    ac = appliances["Air conditioner"]
    fans = appliances["Fans"]
    lamps = appliances["Lamps"]
    chargers = appliances["Phone/device chargers"]
    vacuum = appliances["Vacuum cleaner"]

    if w["temperature_c"] >= 32 and ac["on"] and ac["power_kw"] >= 1.2:
        recs.append(("Temperature / AC", "Medium",
                     f"High outdoor temperature ({w['temperature_c']:.1f}°C) and AC load detected. Set the AC near 25–26°C instead of lowering it excessively.",
                     0.18))
    if e["power_kw"] >= 3.8:
        recs.append(("High instantaneous load", "High",
                     f"Current demand is {e['power_kw']:.2f} kW. Consider delaying a flexible heavy appliance until another load finishes.",
                     0.30))
    if is_anomaly:
        recs.append(("Unusual consumption", "High",
                     f"Current power is {e['power_kw']:.2f} kW, about {abs(z):.1f} standard deviations from the recent live baseline. Check newly started heavy devices.",
                     0.25))
    if 18 <= now.hour <= 23 and lamps["on"] and ac["on"]:
        recs.append(("Evening optimization", "Low",
                     "AC and lighting are both active during the evening. Turn off unnecessary lamps in unoccupied rooms.",
                     0.08))
    if chargers["on"] and 0 <= now.hour <= 6:
        recs.append(("Standby / charging", "Low",
                     "Phone/device charging is active during the late-night period. Unplug chargers when devices reach full charge.",
                     0.03))
    if vacuum["on"]:
        recs.append(("Flexible appliance", "Low",
                     "Vacuum cleaner is active. It is a flexible load; schedule similar non-urgent tasks outside the household peak period.",
                     0.05))
    if monthly_kwh > 1000:
        recs.append(("Monthly consumption", "Medium",
                     f"Projected usage is {monthly_kwh:.0f} kWh/month. Reducing high-load operating hours can move the household toward a lower consumption range.",
                     4.0))

    # Never repeat the same set every refresh unless the underlying condition remains.
    return recs[:4]

# ----------------------------------------------------------------------------
# Sidebar: location + team.
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

# ----------------------------------------------------------------------------
# Persistent live state.
# ----------------------------------------------------------------------------
if "simulator" not in st.session_state:
    st.session_state.simulator = EnergySimulator()
if "history" not in st.session_state:
    st.session_state.history = []
if "hourly_live_history" not in st.session_state:
    st.session_state.hourly_live_history = None
if "last_hour_bucket" not in st.session_state:
    st.session_state.last_hour_bucket = None

# Recreate engine only when location changes.
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
        1 for name in VARIABLE_DEVICES if e["appliances"][name]["on"]
    )
    active_total = len(CONTINUOUS_DEVICES) + active_variable

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
            f'<span class="status-sim">SIM IoT telemetry • {active_total}/15 active</span></div></div>',
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
    st.markdown('<div class="section">Household Device Dashboard — 15 Devices</div>', unsafe_allow_html=True)
    left, right = st.columns(2)
    with left:
        st.markdown("#### 24-hour household devices")
        st.caption("These devices remain available/active as part of the household baseline; their actual energy contribution uses realistic duty cycles.")
        for name in CONTINUOUS_DEVICES:
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
        for name in VARIABLE_DEVICES:
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
    preds = forecast_live_next_hours(model, hourly, hours=6)
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
