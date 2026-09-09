def generate_recommendations(high_night_usage=False, high_sm3=False, repeated_peak=False, current_kw=0, hour=0, sub_metering_pct=None):
    recommendations = []
    if high_night_usage or (hour in [1,2,3,4,5] and current_kw > 0.8):
        recommendations.append(f"Idle consumption {current_kw:.2f}kW at {hour}h (01-05h rule from notebook 05). Consider checking standby: router, fridge, TV standby - saving {(current_kw-0.3):.2f} kWh")
    if high_sm3 or (sub_metering_pct and sub_metering_pct > 40):
        recommendations.append("Sub-metering 3 (AC/water-heater) is 72.8% of total per EDA. Consider scheduling AC to 26°C, shifting to off-peak 13-16h - saving ~20% tariff")
    if repeated_peak or hour in [18,19,20,21]:
        recommendations.append(f"High consumption {current_kw:.2f}kW at peak hour {hour}h (top 3 peaks are 20h,21h,19h per analysis). Shift washing/dishwasher to off-peak - saving {current_kw*0.2:.2f} kWh")
    return recommendations
