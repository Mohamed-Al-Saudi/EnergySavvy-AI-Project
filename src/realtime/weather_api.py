import requests


DEFAULT_LOCATION = {
    "latitude": 30.0444,
    "longitude": 31.2357,
    "city": "Cairo",
}


def get_location_by_ip() -> dict:
    """
    Estimate the user's location using their public IP address.

    Returns a dictionary containing latitude, longitude, and city.
    Falls back to Cairo if the service is unavailable or returns
    invalid data.
    """

    try:
        response = requests.get(
            "https://ipapi.co/json/",
            timeout=5
        )
        response.raise_for_status()

        data = response.json()

        latitude = data.get("latitude")
        longitude = data.get("longitude")
        city = data.get("city")

        if latitude is None or longitude is None:
            raise ValueError("Invalid location data")

        return {
            "latitude": float(latitude),
            "longitude": float(longitude),
            "city": city or DEFAULT_LOCATION["city"],
        }

    except (requests.RequestException, ValueError, TypeError):
        return DEFAULT_LOCATION.copy()


def get_current_weather(latitude: float, longitude: float) -> dict:
    """
    Get current weather conditions from Open-Meteo.

    Falls back to safe default values if the weather API
    is unavailable.
    """

    url = "https://api.open-meteo.com/v1/forecast"

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "current": "temperature_2m,relative_humidity_2m",
        "timezone": "auto",
    }

    try:
        response = requests.get(
            url,
            params=params,
            timeout=10
        )
        response.raise_for_status()

        data = response.json()
        current = data.get("current", {})

        return {
            "temperature_c": current.get("temperature_2m", 30.0),
            "humidity_percent": current.get(
                "relative_humidity_2m",
                60.0
            ),
            "latitude": latitude,
            "longitude": longitude,
        }

    except (requests.RequestException, ValueError, TypeError):
        return {
            "temperature_c": 30.0,
            "humidity_percent": 60.0,
            "latitude": latitude,
            "longitude": longitude,
        }
