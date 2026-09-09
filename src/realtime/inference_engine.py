from pathlib import Path
from collections import deque
from datetime import datetime

import joblib
import pandas as pd
import numpy as np

from src.models.anomaly_detection import add_anomaly_flag
from src.recommendations.recommendation_engine import generate_recommendations


class InferenceEngine:
    """
    Production inference layer for EnergySavvy AI.

    Converts real-time household measurements into the feature
    structure expected by the trained forecasting model and
    generates:
        1. Consumption forecast
        2. Anomaly detection
        3. Data-driven recommendations
    """

    FORECAST_FEATURES = [
        "hour",
        "dayofweek",
        "month",
        "is_weekend",
        "lag_1",
        "lag_2",
        "lag_3",
        "lag_24",
        "lag_168",
        "roll_24_mean",
        "roll_24_std",
    ]

    def __init__(self, model_path="models/forecast_rf.pkl"):
        self.model_path = Path(model_path)

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Forecast model not found: {self.model_path}"
            )

        self.forecast_model = joblib.load(self.model_path)

        # Store recent instantaneous power measurements.
        # 168 = one week of hourly observations.
        self.power_history = deque(maxlen=168)

    def add_measurement(self, power_kw: float) -> None:
        """Add a new real-time power measurement."""

        if power_kw < 0:
            raise ValueError("Power cannot be negative.")

        self.power_history.append(float(power_kw))

    def _build_forecast_features(self, timestamp: datetime):
        """
        Build the exact 11 features used by the trained RF model.
        """

        if len(self.power_history) < 168:
            return None

        values = list(self.power_history)

        features = {
            "hour": timestamp.hour,
            "dayofweek": timestamp.weekday(),
            "month": timestamp.month,
            "is_weekend": int(timestamp.weekday() >= 5),

            "lag_1": values[-1],
            "lag_2": values[-2],
            "lag_3": values[-3],
            "lag_24": values[-24],
            "lag_168": values[-168],

            "roll_24_mean": float(np.mean(values[-24:])),
            "roll_24_std": float(np.std(values[-24:])),
        }

        return pd.DataFrame(
            [features],
            columns=self.FORECAST_FEATURES,
        )

    def predict_consumption(self, timestamp: datetime) -> dict:
        """
        Predict the next consumption value.

        The model cannot produce a valid prediction until enough
        real-time history has been collected.
        """

        features = self._build_forecast_features(timestamp)

        if features is None:
            return {
                "available": False,
                "prediction_kw": None,
                "message": (
                    f"Collecting data: "
                    f"{len(self.power_history)}/168 measurements"
                ),
            }

        prediction = float(
            self.forecast_model.predict(features)[0]
        )

        return {
            "available": True,
            "prediction_kw": round(max(0.0, prediction), 3),
            "message": "Forecast generated successfully.",
        }

    def detect_anomaly(self, current_kw: float) -> dict:
        """
        Detect unusual instantaneous consumption.

        Uses the live history to establish a dynamic baseline.
        """

        if len(self.power_history) < 24:
            return {
                "available": False,
                "is_anomaly": False,
                "score": None,
                "message": (
                    f"Collecting baseline: "
                    f"{len(self.power_history)}/24 measurements"
                ),
            }

        history = np.array(list(self.power_history)[-24:])

        mean_power = float(np.mean(history))
        std_power = float(np.std(history))

        # Avoid division by zero when consumption is almost constant.
        if std_power < 0.01:
            std_power = 0.01

        z_score = (current_kw - mean_power) / std_power

        is_anomaly = abs(z_score) >= 3.0

        return {
            "available": True,
            "is_anomaly": bool(is_anomaly),
            "score": round(float(z_score), 3),
            "baseline_kw": round(mean_power, 3),
            "message": (
                "Unusual consumption detected."
                if is_anomaly
                else "Consumption is within the normal range."
            ),
        }

    def generate_live_recommendations(
        self,
        current_kw: float,
        timestamp: datetime,
        anomaly: dict,
        appliances: dict | None = None,
    ) -> list:
        """
        Generate recommendations from current real-time conditions.
        """

        hour = timestamp.hour

        high_night_usage = (
            hour in [1, 2, 3, 4, 5]
            and current_kw > 0.8
        )

        high_sm3 = False
        repeated_peak = (
            hour in [18, 19, 20, 21]
            and current_kw > 1.5
        )

        recommendations = generate_recommendations(
            high_night_usage=high_night_usage,
            high_sm3=high_sm3,
            repeated_peak=repeated_peak,
            current_kw=current_kw,
            hour=hour,
        )

        # Add anomaly-specific recommendation.
        if anomaly.get("is_anomaly"):
            recommendations.insert(
                0,
                (
                    f"Unusual consumption detected at "
                    f"{current_kw:.2f} kW. "
                    "Check simultaneously operating appliances."
                ),
            )

        # Appliance-aware recommendation.
        if appliances:
            high_power_appliances = [
                name
                for name, data in appliances.items()
                if data.get("on") and data.get("power_kw", 0) >= 1.0
            ]

            if high_power_appliances:
                names = ", ".join(
                    high_power_appliances
                )

                recommendations.append(
                    f"High-power appliances currently active: "
                    f"{names}. Consider switching off unnecessary loads."
                )

        return recommendations

    def process(self, live_data: dict) -> dict:
        """
        Process one complete real-time measurement.

        Returns all AI outputs required by the dashboard.
        """

        energy = live_data["energy"]

        current_kw = float(
            energy["power_kw"]
        )

        timestamp = datetime.fromisoformat(
            energy["timestamp"]
        )

        self.add_measurement(current_kw)

        forecast = self.predict_consumption(
            timestamp
        )

        anomaly = self.detect_anomaly(
            current_kw
        )

        recommendations = self.generate_live_recommendations(
            current_kw=current_kw,
            timestamp=timestamp,
            anomaly=anomaly,
            appliances=energy.get("appliances"),
        )

        return {
            "forecast": forecast,
            "anomaly": anomaly,
            "recommendations": recommendations,
        }
