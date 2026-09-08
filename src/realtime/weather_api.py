import requests


def get_current_weather(latitude: float, longitude: float) -> dict:
    """
    Get current weather from Open-Meteo.

    Returns a simple dictionary that the dashboard/realtime
    system can use without knowing anything about the API.
    """

    url = "https://api.open-meteo.com/v1/forecast"

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "current": [
            "temperature_2m",
            "relative_humidity_2m",
            "apparent_temperature",
            "precipitation",
            "cloud_cover",
            "wind_speed_10m",
        ],
        "timezone": "auto",
    }

    response = requests.get(url, params=params, timeout=10)
    response.raise_for_status()

    data = response.json()
    current = data["current"]

    return {
        "timestamp": current["time"],
        "temperature_c": current["temperature_2m"],
        "humidity_percent": current["relative_humidity_2m"],
        "apparent_temperature_c": current["apparent_temperature"],
        "precipitation_mm": current["precipitation"],
        "cloud_cover_percent": current["cloud_cover"],
        "wind_speed_kmh": current["wind_speed_10m"],
    }
