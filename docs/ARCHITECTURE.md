# Repository Architecture

```text
EnergySavvy-AI/
├── data/
│   ├── household_power/
│   │   ├── raw/          # Original UCI dataset
│   │   └── processed/    # Cleaned/engineered outputs
│   └── cairo_weather/
│       ├── raw/          # Original Cairo weather dataset
│       └── processed/    # Cleaned weather outputs
├── notebooks/            # Exploratory and experimental work
├── src/                  # Reusable Python application code
├── models/               # Saved trained models
├── dashboard/            # Streamlit application
├── reports/              # Figures and evaluation results
├── tests/                # Automated tests
└── docs/                 # Project documentation
```

## Data separation

The two datasets are intentionally separated throughout the repository:

- Household modules work with electricity consumption.
- Weather modules work with Cairo weather.
- No merge module currently exists.

This structure makes future integration possible without making unsupported assumptions today.
