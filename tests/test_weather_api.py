from src.realtime.weather_api import get_current_weather


def test_weather_api():
    weather = get_current_weather(30.0444, 31.2357)

    assert isinstance(weather, dict)
    assert "temperature_c" in weather
    assert "humidity_percent" in weather
    assert "wind_speed_kmh" in weather
