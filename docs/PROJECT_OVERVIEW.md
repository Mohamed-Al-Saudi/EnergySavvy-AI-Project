# Project Overview

## What is EnergySavvy AI?

EnergySavvy AI is a software-based intelligent system for analyzing household electricity consumption. It transforms raw energy data into useful information for users through:

- Consumption pattern analysis
- Future consumption forecasting
- Unusual usage detection
- Data-driven recommendations
- Interactive visualization

## Real-world operation

A deployed version may receive electricity information from a smart meter, an API, a database, or a CSV upload. The current prototype focuses on CSV-based historical data.

## Current prototype input

The UCI household dataset is the primary source for:

- `global_active_power`
- `global_reactive_power`
- `voltage`
- `global_intensity`
- `sub_metering_1`
- `sub_metering_2`
- `sub_metering_3`
- Date and time information

## Cairo Weather dataset

The Cairo weather dataset is intentionally stored and analyzed separately. It may support future localized deployment when matching Cairo electricity consumption data becomes available.

## Important scientific rule

Do not directly merge Cairo weather observations with the French household electricity observations simply because dates overlap. A future integration requires weather and electricity measurements representing the same location and compatible time periods.
