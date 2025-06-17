import joblib
import pandas as pd
from datetime import datetime
import sys

# Load model and encoders
model_data = joblib.load('crowd_prediction_model.pkl')
model = model_data['model']
encoders = model_data['label_encoders']

# Helper to get season
def get_season(month):
    if month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    elif month in [9, 10, 11]:
        return 'Fall'
    else:
        return 'Winter'

# Example input (can be replaced with user input or arguments)
place_id = int(sys.argv[1]) if len(sys.argv) > 1 else 1
category = sys.argv[2] if len(sys.argv) > 2 else 'Temple'
district = sys.argv[3] if len(sys.argv) > 3 else 'Kathmandu'
time_slot = sys.argv[4] if len(sys.argv) > 4 else 'morning'
date_str = sys.argv[5] if len(sys.argv) > 5 else datetime.now().strftime('%Y-%m-%d')
date = datetime.strptime(date_str, '%Y-%m-%d')
day_of_week = date.weekday()
month = date.month
season = get_season(month)
is_weekend = 1 if day_of_week >= 5 else 0
is_holiday = 0  # You can add logic for holidays
weather_condition = sys.argv[6] if len(sys.argv) > 6 else 'Sunny'

# Encode features
features = [
    place_id,
    encoders['category'].transform([category])[0],
    encoders['district'].transform([district])[0],
    encoders['time_slot'].transform([time_slot])[0],
    day_of_week,
    month,
    encoders['season'].transform([season])[0],
    is_weekend,
    is_holiday,
    encoders['weather_condition'].transform([weather_condition])[0]
]

# Predict
pred = model.predict([features])[0]
print(f'Predicted crowd level for place_id={place_id}, category={category}, district={district}, time_slot={time_slot}, date={date_str}, weather={weather_condition}: {round(pred, 1)}%') 