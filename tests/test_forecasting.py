from src.models.forecasting import evaluate_forecast

def test_forecast_metrics():
    metrics = evaluate_forecast([1, 2], [1, 2])
    assert metrics["mae"] == 0.0
    assert metrics["rmse"] == 0.0
