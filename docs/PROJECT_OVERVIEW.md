# Project Overview

## What is EnergySavvy AI?

EnergySavvy AI is a software-based intelligent energy management system that analyzes household electricity consumption to:

- Understand consumption patterns
- Forecast future consumption
- Detect unusual behavior
- Generate data-driven energy-saving recommendations
- Provide an interactive real-time dashboard

The system is designed to demonstrate how artificial intelligence can support energy conservation and rationalization.

---

## System Data Strategy

EnergySavvy AI separates **historical training data** from **real-time application data**.

### Historical training data

The project uses:

- UCI Individual Household Electric Power Consumption
- Historical Cairo weather data

These datasets are used for:

- Data analysis
- Feature engineering
- Model training
- Model testing and evaluation
- Developing forecasting and anomaly-detection approaches

They are **not used as live dashboard inputs**.

### Real-time application data

The deployed prototype uses:

- A **weather API** for current localized weather information.
- A **software-based energy simulator** for continuously generated electricity consumption data.

The energy simulator is a temporary replacement for physical IoT/smart-meter sensors.

---

## Real-Time Operation

The deployed application follows this general flow:

```text
User Location
     |
     v
Weather API ───────────────┐
                           |
Energy Consumption         |
Simulator ────────────────>|
                           v
                    Real-Time Data
                           |
                           v
                    AI Inference
                    /     |      \
                   /      |       \
                  v       v        v
             Forecast  Anomaly  Recommendations
                  \      |       /
                   \     |      /
                    v    v      v
                    Live Dashboard
```

The dashboard can display:

- Current location
- Current temperature and humidity
- Instantaneous power
- Voltage and current
- Appliance states
- Future consumption forecast
- Anomaly status
- Energy-saving recommendations
- Estimated monthly electricity cost

---

## Current Prototype Inputs

### Historical electricity dataset

The UCI dataset provides historical variables such as:

- `Global_active_power`
- `Global_reactive_power`
- `Voltage`
- `Global_intensity`
- `Sub_metering_1`
- `Sub_metering_2`
- `Sub_metering_3`
- Date and time

These variables support offline model development and testing.

### Historical weather dataset

The Cairo weather dataset is kept separate from the historical electricity dataset.

It is used for independent weather analysis and development rather than being presented as live weather.

### Live weather

Current weather is retrieved from the weather API according to the detected or selected location.

### Simulated electricity

The energy simulator generates current household electricity behavior continuously.

It includes multiple appliance categories and produces instantaneous power, voltage, current, appliance states, and timestamps.

---

## Future IoT Integration

The simulator is intentionally designed as a temporary input layer.

When physical smart meters or IoT sensors become available:

```text
Current:
Energy Simulator
       |
       v
Real-Time AI Pipeline

Future:
IoT / Smart Meter
       |
       v
Real-Time AI Pipeline
```

The AI inference and dashboard layers can therefore remain largely unchanged.

---

## Scientific Data Rule

Historical electricity and historical Cairo weather data should not be directly merged simply because their dates overlap.

The original electricity dataset represents a different location and historical period from the Cairo weather dataset.

Any future combined model should use electricity and weather observations representing the same location and compatible time periods.

The current production system avoids this unsupported merge by obtaining **current weather independently through the weather API** and using **simulated real-time electricity data**.
