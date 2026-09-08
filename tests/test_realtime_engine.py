from unittest.mock import patch

from src.realtime.realtime_engine import RealtimeEngine


def test_realtime_engine_manual_location():
    """Test the engine using a manually specified location."""

    fake_weather = {
        "temperature_c": 32.0,
        "humidity_percent": 60.0,
        "latitude": 30.0444,
        "longitude": 31.2357,
    }

    with patch(
        "src.realtime.realtime_engine.get_current_weather",
        return_value=fake_weather,
    ):
        engine = RealtimeEngine(
            latitude=30.0444,
            longitude=31.2357,
            auto_location=False,
        )

        data = engine.get_live_data()

    assert isinstance(data, dict)

    assert "timestamp" in data
    assert "weather" in data
    assert "energy" in data
    assert "location" in data

    assert data["weather"]["temperature_c"] == 32.0
    assert data["weather"]["humidity_percent"] == 60.0

    assert "voltage_v" in data["energy"]
    assert "current_a" in data["energy"]
    assert "power_kw" in data["energy"]
    assert "appliances" in data["energy"]

    assert data["location"]["city"] == "Cairo"
    assert data["location"]["lat"] == 30.0444
    assert data["location"]["lon"] == 31.2357


def test_realtime_engine_auto_location():
    """Test automatic location detection."""

    fake_location = {
        "latitude": 30.0444,
        "longitude": 31.2357,
        "city": "Cairo",
    }

    fake_weather = {
        "temperature_c": 32.0,
        "humidity_percent": 60.0,
        "latitude": 30.0444,
        "longitude": 31.2357,
    }

    with patch(
        "src.realtime.realtime_engine.get_location_by_ip",
        return_value=fake_location,
    ):
        with patch(
            "src.realtime.realtime_engine.get_current_weather",
            return_value=fake_weather,
        ):
            engine = RealtimeEngine(auto_location=True)
            data = engine.get_live_data()

    assert engine.latitude == 30.0444
    assert engine.longitude == 31.2357
    assert engine.city == "Cairo"

    assert data["location"]["city"] == "Cairo"
