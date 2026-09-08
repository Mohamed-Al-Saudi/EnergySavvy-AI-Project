from src.realtime.weather_api import (
    get_current_weather,
    get_location_by_ip,
)
from src.realtime.energy_simulator import EnergySimulator


DEFAULT_LATITUDE = 30.0444
DEFAULT_LONGITUDE = 31.2357
DEFAULT_CITY = "Cairo"


class RealtimeEngine:
    """
    Central real-time engine for EnergySavvy AI.

    Combines:
    - User/location information
    - Live weather data
    - Simulated household electricity data

    The EnergySimulator can later be replaced by real IoT sensors.
    """

    def __init__(
        self,
        latitude: float | None = None,
        longitude: float | None = None,
        auto_location: bool = True,
    ):
        if auto_location and latitude is None and longitude is None:
            location = get_location_by_ip()

            self.latitude = location["latitude"]
            self.longitude = location["longitude"]
            self.city = location["city"]

        else:
            self.latitude = (
                DEFAULT_LATITUDE
                if latitude is None
                else latitude
            )

            self.longitude = (
                DEFAULT_LONGITUDE
                if longitude is None
                else longitude
            )

            self.city = DEFAULT_CITY

        self.simulator = EnergySimulator()

    def get_live_data(self) -> dict:
        """
        Collect the latest weather and household energy data.
        """

        weather = get_current_weather(
            self.latitude,
            self.longitude,
        )

        energy = self.simulator.generate(
            temperature=weather["temperature_c"]
        )

        return {
            "timestamp": energy["timestamp"],
            "weather": weather,
            "energy": energy,
            "location": {
                "city": self.city,
                "lat": self.latitude,
                "lon": self.longitude,
            },
        }

    def update_location(
        self,
        latitude: float,
        longitude: float,
        city: str | None = None,
    ) -> None:
        """
        Update the location used by the real-time engine.
        """

        self.latitude = latitude
        self.longitude = longitude

        if city is not None:
            self.city = city
