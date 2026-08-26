# ⚡ EnergySavvy AI

## Intelligent Energy Management for Sustainability

EnergySavvy AI is a software-based intelligent energy management system that analyzes household electricity consumption data to understand usage patterns, forecast future consumption, detect unusual behavior, and generate data-driven recommendations.

## Current data strategy

This repository currently contains **two separate data domains**:

1. **UCI Household Electric Power Consumption**
   - Primary dataset for consumption analysis, forecasting, anomaly detection, and recommendations.
2. **Cairo Weather**
   - Separate contextual dataset for weather analysis and future localized integration.

> Important: The Cairo weather dataset is **not merged** with the UCI household dataset because they represent different locations and are not directly comparable.

## System workflow

```text
Input Data
   |
   +--> UCI Household Power ---> Cleaning ---> Features ---> Forecasting
   |                                                    \-> Anomaly Detection
   |                                                             |
   |                                                             v
   |                                                     Recommendations
   |
   +--> Cairo Weather ----------> Cleaning ---> Weather Analysis
                                                               |
                                                               v
                                                         Dashboard Views
```

See `docs/PROJECT_OVERVIEW.md` for the full explanation.
