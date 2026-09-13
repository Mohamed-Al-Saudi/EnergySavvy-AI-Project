import random
from datetime import datetime


class EnergySimulator:
    """
    Simulates real-time household electricity measurements.

    This is a temporary replacement for future IoT sensors.
    It generates realistic appliance-level power consumption.
    """

    def __init__(self):
        self.appliances = {
            "refrigerator": {
                "rated_power_kw": 0.15,
                "probability": 0.95,
            },
            "air_conditioner": {
                "rated_power_kw": 1.20,
                "probability": 0.30,
            },
            "television": {
                "rated_power_kw": 0.08,
                "probability": 0.25,
            },
            "lights": {
                "rated_power_kw": 0.10,
                "probability": 0.30,
            },
            "laptop": {
                "rated_power_kw": 0.06,
                "probability": 0.20,
            },
            "washing_machine": {
                "rated_power_kw": 0.50,
                "probability": 0.05,
            },
            "water_heater": {
                "rated_power_kw": 1.50,
                "probability": 0.10,
            },
            "microwave": {
                "rated_power_kw": 1.00,
                "probability": 0.03,
            },
        }

    def _get_probability(self, appliance, hour, temperature):
        """
        Adjust appliance probability according to time and temperature.
        """

        probability = self.appliances[appliance]["probability"]

        # Evening → more household activity
        if 18 <= hour <= 23:
            if appliance in ["television", "lights", "laptop"]:
                probability += 0.30

        # Night → fewer appliances
        if 0 <= hour < 6:
            if appliance in [
                "television",
                "lights",
                "laptop",
                "washing_machine",
                "microwave",
            ]:
                probability *= 0.2

        # Hot weather → AC more likely
        if appliance == "air_conditioner":
            if temperature >= 32:
                probability = 0.85
            elif temperature >= 28:
                probability = 0.60
            elif temperature >= 24:
                probability = 0.30
            else:
                probability = 0.05

        return min(probability, 1.0)

    def generate(self, temperature=30.0):
        """
        Generate one instantaneous household measurement.

        Returns:
            dict containing voltage, current, total power,
            and appliance-level measurements.
        """

        now = datetime.now()
        hour = now.hour

        appliance_data = {}
        total_power_kw = 0.0

        for appliance, config in self.appliances.items():

            probability = self._get_probability(
                appliance,
                hour,
                temperature
            )

            is_on = random.random() < probability

            if is_on:
                rated_power = config["rated_power_kw"]

                # Small variation to avoid perfectly constant readings
                actual_power = rated_power * random.uniform(0.90, 1.10)

            else:
                actual_power = 0.0

            appliance_data[appliance] = {
                "on": is_on,
                "power_kw": round(actual_power, 3),
            }

            total_power_kw += actual_power

        # Simulated household voltage
        voltage = random.uniform(220, 240)

        # Simplified current calculation
        current = (total_power_kw * 1000) / voltage

        return {
            "timestamp": now.isoformat(),
            "voltage_v": round(voltage, 2),
            "current_a": round(current, 2),
            "power_kw": round(total_power_kw, 3),
            "appliances": appliance_data,
        }
