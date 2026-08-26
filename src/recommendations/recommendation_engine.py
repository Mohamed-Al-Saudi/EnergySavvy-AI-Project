"""Rule-based recommendation layer.

Recommendations should be traceable to observed conditions and should use
careful language such as 'consider checking' rather than claiming certainty.
"""

def generate_recommendations(high_night_usage=False, high_sm3=False, repeated_peak=False):
    recommendations = []

    if high_night_usage:
        recommendations.append(
            "Unusual nighttime consumption was observed. Consider checking devices that may remain active during low-activity hours."
        )

    if high_sm3:
        recommendations.append(
            "Consumption in the water-heater/AC sub-metering group is higher than the defined normal pattern. Consider reviewing appliance schedules in this group."
        )

    if repeated_peak:
        recommendations.append(
            "Repeated high consumption occurs during a similar period. Reviewing appliance usage during that period may identify opportunities to reduce unnecessary consumption."
        )

    return recommendations
