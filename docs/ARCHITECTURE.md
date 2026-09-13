# Repository Architecture

```text
EnergySavvy-AI/
├── data/
│   ├── household_power/
│   │   ├── raw/              # Original UCI historical electricity dataset
│   │   └── processed/        # Cleaned/engineered training outputs
│   └── cairo_weather/
│       ├── raw/              # Original historical Cairo weather dataset
│       └── processed/        # Cleaned weather outputs for analysis/training
├── notebooks/                # Exploratory, analysis, and model-development work
├── src/
│   ├── data/                 # Data preprocessing utilities
│   ├── features/             # Household and weather feature engineering
│   ├── models/               # Forecasting and anomaly-detection logic
│   ├── recommendations/      # Recommendation generation
│   └── realtime/              # Live weather API and simulated energy data
├── models/                   # Saved trained AI models
├── dashboard/                # Streamlit application
├── reports/                  # Evaluation results and generated reports
├── tests/                    # Automated tests
└── docs/                     # Project documentation
```

## Data architecture

EnergySavvy AI separates **historical training data** from **real-time application data**.

### Historical data

The historical datasets are used for:

- Exploratory data analysis
- Feature engineering
- AI model training
- Model testing and evaluation
- Understanding historical consumption patterns

The UCI household electricity dataset and historical Cairo weather dataset are **not treated as live data by the deployed dashboard**.

### Real-time / production data

The deployed prototype uses two live/production inputs:

1. **Weather API**
   - Retrieves current weather conditions according to the detected or selected location.
   - Provides current information such as temperature, humidity, wind speed, and location.

2. **Energy consumption simulator**
   - Continuously generates simulated household electricity readings.
   - Represents the temporary software replacement for real IoT/smart-meter sensors.
   - Generates instantaneous power, voltage, current, appliance states, and timestamps.
   - Can later be replaced by real IoT energy sensors without changing the overall system architecture.

## Data separation

The historical electricity and historical weather datasets remain intentionally separated:

- Household modules work with historical electricity consumption.
- Weather modules work with historical weather data.
- Historical electricity and historical Cairo weather are not directly merged.
- Real-time weather is obtained independently through the weather API.
- Real-time electricity data is generated independently by the simulator.

The production flow is:

```text
Historical Electricity Data ──> Preprocessing ──> Feature Engineering
                                             │
                                             v
                                      AI Model Training
                                             │
                                             v
                                      Saved AI Models
                                             │
                                             v
                         ┌───────────────────┴───────────────────┐
                         │           Deployed Dashboard          │
                         │                                       │
                         │  Weather API ──────┐                 │
                         │                    ├─> Real-Time     │
                         │  Energy Simulator ─┘    AI Pipeline   │
                         │                                       │
                         │          Forecasting                 │
                         │          Anomaly Detection            │
                         │          Recommendations              │
                         └───────────────────────────────────────┘
```

This structure allows the prototype to demonstrate real-time behavior while keeping historical datasets strictly within the training and development pipeline.
