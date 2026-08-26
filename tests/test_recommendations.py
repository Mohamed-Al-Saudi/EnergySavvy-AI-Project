from src.recommendations.recommendation_engine import generate_recommendations

def test_recommendations_are_generated():
    result = generate_recommendations(high_night_usage=True)
    assert len(result) == 1
