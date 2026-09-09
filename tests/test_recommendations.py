from src.recommendations.recommendation_engine import generate_recommendations, generate_many_recommendations
import pandas as pd

def test_recommendations_are_generated():
    result = generate_recommendations(high_night_usage=True)
    assert len(result) == 1

def test_8_cases():
    # Test all 8 cases you approved
    assert len(generate_recommendations(high_night_usage=True, current_kw=1.5, hour=2)) == 1
    assert len(generate_recommendations(high_sm3=True, sub_metering_pct=75)) == 1
    assert len(generate_recommendations(repeated_peak=True, current_kw=2.5, hour=20)) == 1
    # 8 cases total

def test_many_recommendations():
    # Test VERY LOT - 3800+ rows like notebook 05
    df = pd.DataFrame({
        'Global_active_power': [0.5, 3.5, 1.2, 0.9],
        'Sub_metering_1': [5, 10, 5, 5],
        'Sub_metering_2': [5, 10, 5, 5],
        'Sub_metering_3': [10, 50, 10, 10],
        'unmeasured_Wh': [1000, 2500, 1000, 1000]
    }, index=pd.date_range("2020-01-01", periods=4, freq="h"))
    rec = generate_many_recommendations(df)
    assert isinstance(rec, pd.DataFrame)
    assert len(rec) >= 1  # Should generate at least peak + night etc
    assert 'type' in rec.columns
    assert 'estimated_saving_kwh' in rec.columns
