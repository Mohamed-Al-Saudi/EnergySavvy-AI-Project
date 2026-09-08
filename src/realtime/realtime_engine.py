from src.realtime.weather_api import get_current_weather
from src.realtime.energy_simulator import EnergySimulator


class RealtimeEngine:
    """
    Combines live weather data with simulated household
    electrical measurements.

    The simulator can later be replaced by real IoT sensors.
    """

    def __init__(self, latitude: float, longitude: float):
        self.latitude = latitude
        self.longitude = longitude

        self.simulator = EnergySimulator()

    def get_live_data(self) -> dict:
        """
        Get the latest complete household state.
        """

        # 1. Get current weather
        weather = get_current_weather(
            self.latitude,
            self.longitude
        )

        # 2. Use current temperature to influence
        #    household energy simulation
        energy = self.simulator.generate(
            temperature=weather["temperature_c"]
        )

        # 3. Combine both sources
        return {
            "timestamp": energy["timestamp"],
            "weather": weather,
            "energy": energy,
        }
