# Development Workflow

## Phase 1: Understand the Historical Data

Run the EDA notebooks separately for:

- Historical UCI household electricity data
- Historical Cairo weather data

These datasets are treated as **training/development data**, not as live application inputs.

## Phase 2: Preprocess

Create reproducible cleaning and preprocessing functions in `src/data/`.

The preprocessing pipeline should produce reliable inputs for model development without modifying the original raw datasets.

## Phase 3: Feature Engineering

Create features required for forecasting and anomaly detection, including:

- Time-based features
- Historical consumption features
- Lag features
- Rolling statistics
- Other validated model features

Weather features can be developed separately from the historical weather dataset.

## Phase 4: Forecasting

Train and compare forecasting models using time-aware validation.

Save the selected trained model in `models/`.

The trained model is then used by the deployed application for inference.

## Phase 5: Anomaly Detection

Develop methods for identifying observations that differ meaningfully from normal consumption behavior.

Validate the approach using historical development data and then apply the inference logic to real-time application readings.

## Phase 6: Recommendations

Translate model outputs and current consumption patterns into cautious, explainable, data-driven recommendations.

Recommendations should be generated from current application conditions rather than displaying historical recommendation tables to the user.

## Phase 7: Real-Time Data Layer

The deployed prototype combines:

1. **Weather API**
   - Current location-aware weather data.

2. **Energy simulator**
   - Continuously generated household electricity readings.
   - Temporary replacement for real IoT/smart-meter data.

The simulator should continuously produce new readings while the application is running.

## Phase 8: Real-Time AI Inference

Use the trained models and real-time inputs to produce:

- Future consumption forecasts
- Anomaly detection results
- Current energy-saving recommendations
- Estimated consumption/cost information

Historical datasets remain behind the training/development boundary.

## Phase 9: Dashboard

Load the trained models and real-time data pipeline into the Streamlit dashboard.

The dashboard should present current information such as:

- Location
- Weather
- Instantaneous power
- Voltage
- Current
- Appliance states
- Forecast
- Anomaly status
- Recommendations
- Estimated electricity cost

## Phase 10: Testing

Run automated tests for:

- Data preprocessing
- Weather API behavior
- Energy simulation
- Forecasting
- Real-time engine
- Other reusable application modules

## Production Data Flow

```text
Historical Data
      |
      v
Preprocessing
      |
      v
Feature Engineering
      |
      v
Model Training & Testing
      |
      v
Saved Models
      |
      +-----------------------------+
                                    |
                                    v
                            Real-Time Application
                              /              \
                             /                \
                            v                  v
                     Weather API       Energy Simulator
                            \                /
                             \              /
                              v            v
                              Real-Time Inputs
                                      |
                                      v
                               AI Inference
                              /      |       \
                             v       v        v
                         Forecast Anomaly Recommendations
                              \      |       /
                               \     |      /
                                v    v      v
                                Dashboard
```

## Rule

Do not move experimental notebook logic into the dashboard until it is stable, tested, and reusable.

Keep historical training data and real-time application data clearly separated.
