"""Live weather + IP location for the deployed dashboard."""
from __future__ import annotations

import math
from datetime import datetime

import requests


# ---------------------------------------------------------------------------
# All 27 Egyptian governorates
# ---------------------------------------------------------------------------
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

DEFAULT_LOCATION = {
    "latitude": GOVERNORATES["Cairo"][0],
    "longitude": GOVERNORATES["Cairo"][1],
    "city": "Cairo",
}


def _weather_code_description(code) -> str:
    mapping = {
        0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
        45: "Fog", 48: "Depositing rime fog",
        51: "Light drizzle", 53: "Moderate drizzle", 55: "Dense drizzle",
        61: "Slight rain", 63: "Moderate rain", 65: "Heavy rain",
        71: "Slight snow", 73: "Moderate snow", 75: "Heavy snow",
        80: "Slight rain showers", 81: "Moderate rain showers", 82: "Violent rain showers",
        95: "Thunderstorm", 96: "Thunderstorm with hail", 99: "Thunderstorm with heavy hail",
    }
    try:
        return mapping.get(int(code), "Current conditions")
    except (TypeError, ValueError):
        return "Current conditions"


def get_ip_location() -> dict:
    """Return city/country/lat/lon for the caller's public IP. Never fails."""
    try:
        r = requests.get("https://ipapi.co/json/", timeout=5)
        r.raise_for_status()
        data = r.json()
        lat = float(data.get("latitude", DEFAULT_LOCATION["latitude"]))
        lon = float(data.get("longitude", DEFAULT_LOCATION["longitude"]))
        return {
            "city": data.get("city") or "Cairo",
            "country": data.get("country_name") or "Egypt",
            "latitude": lat,
            "longitude": lon,
            "source": "IP auto-detection",
        }
    except Exception:
        return {
            "city": "Cairo",
            "country": "Egypt",
            "latitude": DEFAULT_LOCATION["latitude"],
            "longitude": DEFAULT_LOCATION["longitude"],
            "source": "Cairo fallback",
        }


def get_live_weather(lat: float, lon: float) -> dict:
    """Return current weather for (lat, lon). Open-Meteo primary, wttr.in fallback."""
    # 1. Open-Meteo
    try:
        url = (
            "https://api.open-meteo.com/v1/forecast"
            f"?latitude={lat}&longitude={lon}"
            "&current=temperature_2m,relative_humidity_2m,wind_speed_10m,"
            "apparent_temperature,weather_code&timezone=auto"
        )
        cur = requests.get(url, timeout=8).json()["current"]
        return {
            "temperature_c": float(cur["temperature_2m"]),
            "humidity_percent": int(round(cur["relative_humidity_2m"])),
            "wind_speed_kmh": float(cur["wind_speed_10m"]),
            "feels_like_c": float(cur["apparent_temperature"]),
            "weather_desc": _weather_code_description(cur["weather_code"]),
            "latitude": lat,
            "longitude": lon,
            "source": "Open-Meteo live",
            "is_simulated": False,
            "timestamp": datetime.now().isoformat(),
        }
    except Exception:
        pass

    # 2. wttr.in
    try:
        c = requests.get(f"https://wttr.in/{lat},{lon}?format=j1", timeout=8).json()["current_condition"][0]
        return {
            "temperature_c": float(c["temp_C"]),
            "humidity_percent": int(c["humidity"]),
            "wind_speed_kmh": float(c["windspeedKmph"]),
            "feels_like_c": float(c.get("FeelsLikeC", c["temp_C"])),
            "weather_desc": c["weatherDesc"][0]["value"],
            "latitude": lat,
            "longitude": lon,
            "source": "wttr.in live",
            "is_simulated": False,
            "timestamp": datetime.now().isoformat(),
        }
    except Exception:
        pass

    # 3. Fallback estimate
    hour = datetime.now().hour
    temp = 28 + 5 * math.sin((hour - 6) / 24 * 2 * math.pi)
    return {
        "temperature_c": round(temp, 1),
        "humidity_percent": 55,
        "wind_speed_kmh": 12.0,
        "feels_like_c": round(temp + 2, 1),
        "weather_desc": "Fallback estimate",
        "latitude": lat,
        "longitude": lon,
        "source": "Fallback estimate (API unavailable)",
        "is_simulated": True,
        "timestamp": datetime.now().isoformat(),
    }


# ---------------------- Legacy aliases (kept for old code) -----------------
def get_location_by_ip() -> dict:
    return get_ip_location()


def get_current_weather(latitude: float, longitude: float) -> dict:
    return get_live_weather(latitude, longitude)
