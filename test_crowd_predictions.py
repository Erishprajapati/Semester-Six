import joblib
import pandas as pd
from datetime import datetime
import sys
import os
import django

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'mainfolder.settings')
django.setup()

from backend.models import Place

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

def predict_crowd_for_place(place_name, time_slot, is_weekend=False):
    try:
        place = Place.objects.get(name__icontains=place_name)
        
        # Get current date features
        current_time = datetime.now()
        day_of_week = current_time.weekday()
        month = current_time.month
        season = get_season(month)
        is_weekend_flag = 1 if is_weekend else (1 if day_of_week >= 5 else 0)
        is_holiday = 0
        weather_condition = 'Sunny'
        
        # Encode features
        features = [
            place.id,
            encoders['category'].transform([place.category])[0],
            encoders['district'].transform([place.district])[0],
            encoders['time_slot'].transform([time_slot])[0],
            day_of_week,
            month,
            encoders['season'].transform([season])[0],
            is_weekend_flag,
            is_holiday,
            encoders['weather_condition'].transform([weather_condition])[0]
        ]
        
        # Predict
        predicted_crowd = model.predict([features])[0]
        predicted_crowd = float(max(0, min(100, round(predicted_crowd, 1))))
        
        # Determine status
        if predicted_crowd > 70:
            status = 'High'
        elif predicted_crowd > 30:
            status = 'Medium'
        else:
            status = 'Low'
            
        return {
            'place_name': place.name,
            'category': place.category,
            'district': place.district,
            'time_slot': time_slot,
            'crowd_level': predicted_crowd,
            'status': status,
            'is_weekend': is_weekend_flag
        }
        
    except Place.DoesNotExist:
        print(f"Place '{place_name}' not found in database")
        return None

# Test predictions for Pashupatinath Temple and Thamel
places_to_test = ["Pashupatinath Temple", "Thamel"]
time_slots = ["morning", "afternoon", "evening"]

print("=" * 80)
print("CROWD PREDICTIONS FOR KATHMANDU PLACES")
print("=" * 80)

for place_name in places_to_test:
    print(f"\n📍 {place_name.upper()}")
    print("-" * 50)
    
    # Test weekday predictions
    print("📅 WEEKDAY PREDICTIONS:")
    for time_slot in time_slots:
        result = predict_crowd_for_place(place_name, time_slot, is_weekend=False)
        if result:
            emoji = "🌅" if time_slot == "morning" else "☀️" if time_slot == "afternoon" else "🌆"
            status_emoji = "🟢" if result['status'] == 'Low' else "🟡" if result['status'] == 'Medium' else "🔴"
            print(f"  {emoji} {time_slot.capitalize()}: {status_emoji} {result['crowd_level']}% ({result['status']})")
    
    # Test weekend predictions
    print("\n📅 WEEKEND PREDICTIONS:")
    for time_slot in time_slots:
        result = predict_crowd_for_place(place_name, time_slot, is_weekend=True)
        if result:
            emoji = "🌅" if time_slot == "morning" else "☀️" if time_slot == "afternoon" else "🌆"
            status_emoji = "🟢" if result['status'] == 'Low' else "🟡" if result['status'] == 'Medium' else "🔴"
            print(f"  {emoji} {time_slot.capitalize()}: {status_emoji} {result['crowd_level']}% ({result['status']})")

print("\n" + "=" * 80)
print("SUMMARY:")
print("=" * 80)
print("🟢 Low (0-30%): Good time to visit, less crowded")
print("🟡 Medium (31-70%): Moderate crowd, still enjoyable")
print("🔴 High (71-100%): Very crowded, consider visiting at different time")
print("\n💡 TIP: Morning is usually the best time for temples and cultural sites!")
print("💡 TIP: Evening is great for shopping areas like Thamel!") 