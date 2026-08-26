# Development Workflow

## Phase 1: Understand the data
Run the EDA notebooks separately for household power and Cairo weather.

## Phase 2: Preprocess
Create reproducible cleaning functions in `src/data/`.

## Phase 3: Feature engineering
Create time and historical consumption features for forecasting.

## Phase 4: Forecasting
Train and compare forecasting models using time-aware validation.

## Phase 5: Anomaly detection
Identify observations that differ meaningfully from learned or defined normal patterns.

## Phase 6: Recommendations
Translate model outputs and measured patterns into cautious, explainable recommendations.

## Phase 7: Dashboard
Load processed outputs and trained models into Streamlit.

## Rule
Do not move experimental notebook logic into the dashboard until it is stable and reusable.
