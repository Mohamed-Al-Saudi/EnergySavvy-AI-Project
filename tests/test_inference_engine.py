from datetime import datetime

from src.realtime.inference_engine import InferenceEngine


def test_inference_engine():

    engine = InferenceEngine(
        model_path="models/forecast_rf.pkl"
    )

    # Simulate one week of hourly measurements.
    for _ in range(168):
        engine.add_measurement(1.0)

    timestamp = datetime(
        2026,
        9,
        9,
        14,
        0,
        0,
    )

    forecast = engine.predict_consumption(
        timestamp
    )

    assert forecast["available"] is True
    assert forecast["prediction_kw"] is not None

    anomaly = engine.detect_anomaly(
        current_kw=1.0
    )

    assert anomaly["available"] is True
    assert isinstance(
        anomaly["is_anomaly"],
        bool,
    )


def test_inference_engine_process():

    engine = InferenceEngine(
        model_path="models/forecast_rf.pkl"
    )

    live_data = {
        "energy": {
            "power_kw": 1.2,
            "appliances": {
                "refrigerator": {
                    "on": True,
                    "power_kw": 0.15,
                },
                "air_conditioner": {
                    "on": True,
                    "power_kw": 1.2,
                },
            },
            "timestamp": "2026-09-09T14:00:00",
        }
    }

    for _ in range(168):
        engine.add_measurement(1.0)

    result = engine.process(
        live_data
    )

    assert "forecast" in result
    assert "anomaly" in result
    assert "recommendations" in result

    assert isinstance(
        result["recommendations"],
        list,
    )
