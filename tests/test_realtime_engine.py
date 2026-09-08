from src.realtime.realtime_engine import RealtimeEngine


def test_realtime_engine():

    engine = RealtimeEngine(
        latitude=30.0444,
        longitude=31.2357
    )

    data = engine.get_live_data()

    assert isinstance(data, dict)

    assert "timestamp" in data
    assert "weather" in data
    assert "energy" in data

    assert "temperature_c" in data["weather"]
    assert "humidity_percent" in data["weather"]

    assert "voltage_v" in data["energy"]
    assert "current_a" in data["energy"]
    assert "power_kw" in data["energy"]
    assert "appliances" in data["energy"]
