from src.realtime.energy_simulator import EnergySimulator


def test_energy_simulator():

    simulator = EnergySimulator()

    data = simulator.generate(temperature=32)

    assert isinstance(data, dict)

    assert "timestamp" in data
    assert "voltage_v" in data
    assert "current_a" in data
    assert "power_kw" in data
    assert "appliances" in data

    assert data["voltage_v"] >= 220
    assert data["voltage_v"] <= 240

    assert data["power_kw"] >= 0
    assert data["current_a"] >= 0

    assert len(data["appliances"]) == 8
