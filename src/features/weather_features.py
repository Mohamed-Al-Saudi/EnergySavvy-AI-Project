"""Feature engineering for Cairo weather analysis.
These features are intentionally independent from household electricity features.
Based on notebook 06_cairo_weather_eda: 5845 rows 2009-2025, 32 cols, mean 23.03°C
"""
import pandas as pd
import numpy as np

def add_temperature_flags(df: pd.DataFrame, temperature_col: str = "temperature_2m_mean (°C)", threshold: float = 32.0) -> pd.DataFrame:
    result = df.copy()
    result["is_high_temperature"] = (result[temperature_col] >= threshold).astype(int)
    result["is_extreme_heat"] = (result[temperature_col] >= 38.0).astype(int)
    result["is_cold"] = (result[temperature_col] <= 12.0).astype(int)
    return result

def add_cairo_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    """Full 25+ features from Cairo EDA - independent from household"""
    result = df.copy()

    # Auto-detect columns - your CSV has "temperature_2m_mean (°C)" etc
    temp_col = next((c for c in df.columns if "temperature_2m_mean" in c), df.columns[1] if len(df.columns)>1 else df.columns[0])
    humidity_col = next((c for c in df.columns if "humidity" in c.lower()), None)
    wind_col = next((c for c in df.columns if "wind_speed_10m_max" in c or "wind_speed" in c.lower()), None)
    rain_col = next((c for c in df.columns if "rain_sum" in c), None)
    radiation_col = next((c for c in df.columns if "shortwave_radiation" in c), None)

    # Temperature - Cairo mean 23.03°C, 75th 28.6°C, max 37.4°C per your EDA
    result["is_high_temperature"] = (result[temp_col] >= 32.0).astype(int)
    result["is_extreme_heat"] = (result[temp_col] >= 36.0).astype(int)
    result["is_heatwave"] = (result[temp_col] >= 35.0).astype(int)
    result["is_cold"] = (result[temp_col] <= 15.0).astype(int)
    result["temp_range"] = result[temp_col].apply(lambda x: "cold" if x<15 else "mild" if x<25 else "hot" if x<32 else "extreme")

    if "apparent_temperature_mean (°C)" in df.columns:
        result["heat_stress"] = result["apparent_temperature_mean (°C)"] - result[temp_col]
        result["is_high_heat_stress"] = (result["heat_stress"] >= 3).astype(int)

    if humidity_col:
        result["is_high_humidity"] = (result[humidity_col] >= 65).astype(int)
        result["is_low_humidity"] = (result[humidity_col] <= 35).astype(int)
        result["is_comfortable_humidity"] = ((result[humidity_col] >= 40) & (result[humidity_col] <= 60)).astype(int)

    if wind_col:
        result["is_high_wind"] = (result[wind_col] >= 25).astype(int)
        result["is_calm"] = (result[wind_col] <= 10).astype(int)
        result["wind_category"] = result[wind_col].apply(lambda x: "calm" if x<10 else "moderate" if x<20 else "windy" if x<30 else "very_windy")
        result["is_good_for_ventilation"] = ((result[wind_col] >= 8) & (result[wind_col] <= 20)).astype(int)

    if rain_col:
        result["is_rainy"] = (result[rain_col] > 0).astype(int)
        result["is_heavy_rain"] = (result[rain_col] >= 5).astype(int)

    if radiation_col:
        result["is_high_radiation"] = (result[radiation_col] >= 25).astype(int)
        result["is_low_sun"] = (result[radiation_col] <= 12).astype(int)

    if isinstance(result.index, pd.DatetimeIndex):
        result["month"] = result.index.month
        result["season"] = result["month"].map({12:"winter",1:"winter",2:"winter",3:"spring",4:"spring",5:"spring",6:"summer",7:"summer",8:"summer",9:"autumn",10:"autumn",11:"autumn"})
        result["is_summer"] = (result["season"]=="summer").astype(int)
        result["is_winter"] = (result["season"]=="winter").astype(int)
        result["is_peak_summer_month"] = result["month"].isin([7,8]).astype(int)

    if humidity_col:
        result["comfort_index"] = result[temp_col] + 0.1*result[humidity_col]
        result["is_uncomfortable"] = ((result[temp_col] >= 30) & (result[humidity_col] >= 60)).astype(int)

    return result
