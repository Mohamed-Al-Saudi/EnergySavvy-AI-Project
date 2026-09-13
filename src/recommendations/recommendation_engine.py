# =====================================================================
# Live / dashboard layer
# =====================================================================
def generate_live_recommendations(live: dict, is_anomaly: bool, z: float, monthly_kwh: float):
    """Return 0..N live recommendations for the current dashboard state.

    Parameters
    ----------
    live : dict
        The live payload built by the dashboard. It must contain at least:
            live["energy"]["power_kw"]        -> float
            live["energy"]["appliances"]      -> {name: {"on": bool, "power_kw": float, ...}}
            live["weather"]["temperature_c"]  -> float
    is_anomaly : bool
        Current anomaly flag from src.models.anomaly_detection.live_anomaly.
    z : float
        Current z-score returned by live_anomaly.
    monthly_kwh : float
        Projected monthly consumption in kWh (from the bill module).

    Returns
    -------
    list[tuple[str, str, str, float]]
        Each item is (title, severity, message, estimated_saving_kwh).
        severity is one of: "High", "Medium", "Low".
    """
    from datetime import datetime

    energy = live.get("energy", {})
    weather = live.get("weather", {})
    appliances = energy.get("appliances", {}) or {}
    power_kw = float(energy.get("power_kw", 0.0))
    temp_c = float(weather.get("temperature_c", 25.0))
    now = datetime.now()

    def _state(name: str) -> dict:
        """Safe accessor for a device state; defaults to off."""
        return appliances.get(name, {"on": False, "power_kw": 0.0})

    ac = _state("Air conditioner")
    fans = _state("Fans")
    lamps = _state("Lamps")
    chargers = _state("Phone/device chargers")
    vacuum = _state("Vacuum cleaner")
    water_heater = _state("Water heater")
    washing_machine = _state("Washing machine")

    recs: list[tuple[str, str, str, float]] = []

    # --- Temperature / AC -------------------------------------------------
    if temp_c >= 32 and ac.get("on") and float(ac.get("power_kw", 0.0)) >= 1.2:
        recs.append((
            "Temperature / AC",
            "Medium",
            f"High outdoor temperature ({temp_c:.1f}°C) and the air conditioner "
            f"is drawing {ac['power_kw']:.2f} kW. Setting the thermostat near "
            f"25–26°C instead of lowering it further reduces compressor run "
            f"time without sacrificing comfort.",
            0.18,
        ))

    # --- High instantaneous load -----------------------------------------
    if power_kw >= 3.8:
        recs.append((
            "High instantaneous load",
            "High",
            f"Current demand is {power_kw:.2f} kW. Several high-power devices "
            f"are running at once. Delaying one flexible appliance until "
            f"another finishes reduces peak demand and the risk of tripping "
            f"the household breaker.",
            0.30,
        ))

    # --- Anomaly ----------------------------------------------------------
    if is_anomaly:
        recs.append((
            "Unusual consumption",
            "High",
            f"Current power is {power_kw:.2f} kW — about {abs(z):.1f} standard "
            f"deviations from the recent live baseline. Check whether a heavy "
            f"appliance was just started or a device is stuck in a continuous "
            f"cycle.",
            0.25,
        ))

    # --- Evening optimization --------------------------------------------
    if 18 <= now.hour <= 23 and lamps.get("on") and ac.get("on"):
        recs.append((
            "Evening optimization",
            "Low",
            "Air conditioning and lighting are both active during the "
            "evening. Turning off lamps in unoccupied rooms and setting "
            "the AC to a moderate temperature trims the evening peak.",
            0.08,
        ))

    # --- Standby charging at night ---------------------------------------
    if chargers.get("on") and 0 <= now.hour <= 6:
        recs.append((
            "Standby / charging",
            "Low",
            "Phone and device charging is active during the late-night "
            "period. Unplugging chargers once devices are full avoids "
            "unnecessary overnight draw.",
            0.03,
        ))

    # --- Flexible appliance ----------------------------------------------
    if vacuum.get("on"):
        recs.append((
            "Flexible appliance",
            "Low",
            "The vacuum cleaner is running. Tasks like this are flexible "
            "loads — scheduling them outside the household peak period "
            "(13:00–16:00 or after 22:00) reduces tariff cost.",
            0.05,
        ))

    # --- Water heater during peak ----------------------------------------
    if water_heater.get("on") and 18 <= now.hour <= 21 and power_kw >= 2.0:
        recs.append((
            "Water heating during peak",
            "Medium",
            "The water heater is active during the evening peak. Heating "
            "water earlier in the afternoon (off-peak window) supplies the "
            "same hot water at a lower tariff rate.",
            0.12,
        ))

    # --- Washing machine during peak -------------------------------------
    if washing_machine.get("on") and 18 <= now.hour <= 21:
        recs.append((
            "Laundry during peak",
            "Medium",
            "A washing cycle is running during the evening peak. Running "
            "full loads and shifting laundry to the 13:00–16:00 off-peak "
            "window lowers both energy and tariff cost.",
            0.10,
        ))

    # --- High monthly consumption ----------------------------------------
    if monthly_kwh > 1000:
        recs.append((
            "Monthly consumption",
            "Medium",
            f"Projected usage is {monthly_kwh:.0f} kWh/month, which pushes "
            f"the household into a higher tariff bracket. Reducing "
            f"high-load operating hours can move consumption toward a "
            f"lower tier and shrink the bill.",
            4.0,
        ))

    # --- Fans can substitute for AC at mild temps ------------------------
    if 24 <= temp_c < 30 and ac.get("on") and float(ac.get("power_kw", 0.0)) >= 1.0 and not fans.get("on"):
        recs.append((
            "Consider fans instead of AC",
            "Low",
            f"At {temp_c:.1f}°C, ceiling or pedestal fans can often maintain "
            f"comfort at a fraction of the AC's power draw. Running fans "
            f"alongside a higher AC setpoint reduces total consumption.",
            0.15,
        ))

    # Cap at 4 so the dashboard stays focused.
    return recs[:4]
