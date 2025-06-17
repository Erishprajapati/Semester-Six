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

def predict_crowd_for_place(place, time_slot, is_weekend=False):
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
        
    return predicted_crowd, status

def get_status_emoji(status):
    return "🟢" if status == 'Low' else "🟡" if status == 'Medium' else "🔴"

def get_time_emoji(time_slot):
    return "🌅" if time_slot == "morning" else "☀️" if time_slot == "afternoon" else "🌆"

# Get all places from database
all_places = Place.objects.all().order_by('category', 'name')
time_slots = ["morning", "afternoon", "evening"]

print("=" * 100)
print("🏛️  COMPLETE CROWD PREDICTIONS FOR ALL PLACES IN DATABASE")
print("=" * 100)
print(f"📅 Date: {datetime.now().strftime('%B %d, %Y')}")
print(f"⏰ Time: {datetime.now().strftime('%I:%M %p')}")
print("=" * 100)

# Group places by category
categories = {}
for place in all_places:
    if place.category not in categories:
        categories[place.category] = []
    categories[place.category].append(place)

# Process each category
for category in sorted(categories.keys()):
    places = categories[category]
    
    print(f"\n🏷️  {category.upper()} PLACES ({len(places)} places)")
    print("=" * 80)
    
    for place in places:
        print(f"\n📍 {place.name}")
        print(f"   📍 District: {place.district}")
        print(f"   🏷️  Tags: {', '.join([tag.name for tag in place.tags.all()])}")
        print("   " + "-" * 60)
        
        # Weekday predictions
        print("   📅 WEEKDAY:")
        weekday_data = []
        for time_slot in time_slots:
            crowd_level, status = predict_crowd_for_place(place, time_slot, is_weekend=False)
            weekday_data.append((time_slot, crowd_level, status))
            emoji = get_time_emoji(time_slot)
            status_emoji = get_status_emoji(status)
            print(f"      {emoji} {time_slot.capitalize()}: {status_emoji} {crowd_level}% ({status})")
        
        # Weekend predictions
        print("   📅 WEEKEND:")
        weekend_data = []
        for time_slot in time_slots:
            crowd_level, status = predict_crowd_for_place(place, time_slot, is_weekend=True)
            weekend_data.append((time_slot, crowd_level, status))
            emoji = get_time_emoji(time_slot)
            status_emoji = get_status_emoji(status)
            print(f"      {emoji} {time_slot.capitalize()}: {status_emoji} {crowd_level}% ({status})")
        
        # Find best time to visit (lowest crowd level)
        all_data = weekday_data + weekend_data
        best_time = min(all_data, key=lambda x: x[1])
        worst_time = max(all_data, key=lambda x: x[1])
        
        print(f"   💡 BEST TIME: {best_time[0].capitalize()} ({best_time[1]}%)")
        print(f"   ⚠️  AVOID: {worst_time[0].capitalize()} ({worst_time[1]}%)")

# Summary statistics
print("\n" + "=" * 100)
print("📊 SUMMARY STATISTICS")
print("=" * 100)

total_places = len(all_places)
print(f"📈 Total Places: {total_places}")

# Count by category
print(f"\n📂 Places by Category:")
for category in sorted(categories.keys()):
    count = len(categories[category])
    print(f"   {category}: {count} places")

# Count by district
districts = {}
for place in all_places:
    if place.district not in districts:
        districts[place.district] = 0
    districts[place.district] += 1

print(f"\n🗺️  Places by District:")
for district in sorted(districts.keys()):
    count = districts[district]
    print(f"   {district}: {count} places")

# Find places with highest and lowest average crowd
place_averages = []
for place in all_places:
    total_crowd = 0
    count = 0
    for time_slot in time_slots:
        for is_weekend in [False, True]:
            crowd_level, _ = predict_crowd_for_place(place, time_slot, is_weekend)
            total_crowd += crowd_level
            count += 1
    avg_crowd = total_crowd / count
    place_averages.append((place.name, avg_crowd))

# Sort by average crowd
place_averages.sort(key=lambda x: x[1])

print(f"\n🏆 CROWD ANALYSIS:")
print(f"   🟢 Least Crowded Places:")
for i, (name, avg) in enumerate(place_averages[:5]):
    print(f"      {i+1}. {name}: {avg:.1f}% average")

print(f"\n   🔴 Most Crowded Places:")
for i, (name, avg) in enumerate(place_averages[-5:]):
    print(f"      {i+1}. {name}: {avg:.1f}% average")

print("\n" + "=" * 100)
print("💡 VISITING TIPS:")
print("=" * 100)
print("🟢 Low (0-30%): Perfect time to visit, peaceful experience")
print("🟡 Medium (31-70%): Good time, moderate crowd")
print("🔴 High (71-100%): Very crowded, consider alternative times")
print("\n🌅 Morning: Best for temples, cultural sites, and peaceful places")
print("☀️ Afternoon: Good for museums, indoor attractions")
print("🌆 Evening: Best for shopping areas, restaurants, and nightlife")
print("\n📅 Weekdays: Generally less crowded than weekends")
print("📅 Weekends: Higher crowds, especially at popular tourist spots") 