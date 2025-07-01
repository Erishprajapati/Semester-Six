import pandas as pd
import numpy as np
import random
import sys
import os

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.improved_ml_model import ImprovedCrowdPredictionModel

# 1. Load original data
original_csv = 'nepal_tourism_crowd_data.csv'
df = pd.read_csv(original_csv)

# 2. Generate 2000+ synthetic rows
categories = [
    ('Religious', 'Kathmandu'),
    ('Park', 'Lalitpur'),
    ('Market', 'Kathmandu'),
    ('Tourist', 'Bhaktapur'),
    ('Museum', 'Kathmandu')
]
time_slots = ['morning', 'afternoon', 'evening']
weather_options = ['Sunny', 'Rainy', 'Cloudy']
seasons = ['Spring', 'Summer', 'Autumn', 'Winter']

synthetic_rows = []

# --- Realistic Nepali morning crowd logic for Kathmandu places ---
kathmandu_places = [
    {'id': 70, 'name': 'Thamel', 'category': 'Market'},
    {'id': 68, 'name': 'Pashupatinath Temple', 'category': 'Temple'},
    {'id': 130, 'name': 'Kathmandu Durbar Square', 'category': 'Historical'},
    {'id': 67, 'name': 'Swayambhunath Stupa', 'category': 'Temple'},
    {'id': 69, 'name': 'Garden of Dreams', 'category': 'Nature'},
    {'id': 134, 'name': 'Freak Street', 'category': 'Cultural'},
    {'id': 83, 'name': 'Sundarijal', 'category': 'Market'},
]
for p in kathmandu_places:
    if p['category'] in ['Temple', 'Nature', 'Park']:
        # Temples and parks: mostly high
        for _ in range(20):
            synthetic_rows.append({
                'place_id': p['id'], 'category': p['category'], 'district': 'Kathmandu', 'time_slot': 'morning', 'hour': 9,
                'day_of_week': random.randint(0, 6), 'month': random.randint(3, 5), 'day': random.randint(1, 28),
                'season': 'Spring', 'is_weekend': 1, 'is_holiday': 0, 'weather_condition': 'Sunny',
                'crowdlevel': random.uniform(75, 95), 'tourist_season': 'Peak', 'festival_period': 'No',
            })
        for _ in range(10):
            synthetic_rows.append({
                'place_id': p['id'], 'category': p['category'], 'district': 'Kathmandu', 'time_slot': 'morning', 'hour': 9,
                'day_of_week': random.randint(0, 6), 'month': random.randint(3, 5), 'day': random.randint(1, 28),
                'season': 'Spring', 'is_weekend': 0, 'is_holiday': 0, 'weather_condition': 'Sunny',
                'crowdlevel': random.uniform(40, 65), 'tourist_season': 'Peak', 'festival_period': 'No',
            })
        for _ in range(5):
            synthetic_rows.append({
                'place_id': p['id'], 'category': p['category'], 'district': 'Kathmandu', 'time_slot': 'morning', 'hour': 9,
                'day_of_week': random.randint(0, 6), 'month': random.randint(3, 5), 'day': random.randint(1, 28),
                'season': 'Spring', 'is_weekend': 0, 'is_holiday': 0, 'weather_condition': 'Rainy',
                'crowdlevel': random.uniform(10, 25), 'tourist_season': 'Peak', 'festival_period': 'No',
            })
    elif p['category'] in ['Market']:
        # Markets: mostly medium
        for _ in range(15):
            synthetic_rows.append({
                'place_id': p['id'], 'category': p['category'], 'district': 'Kathmandu', 'time_slot': 'morning', 'hour': 9,
                'day_of_week': random.randint(0, 6), 'month': random.randint(3, 5), 'day': random.randint(1, 28),
                'season': 'Spring', 'is_weekend': 1, 'is_holiday': 0, 'weather_condition': 'Sunny',
                'crowdlevel': random.uniform(40, 65), 'tourist_season': 'Peak', 'festival_period': 'No',
            })
        for _ in range(10):
            synthetic_rows.append({
                'place_id': p['id'], 'category': p['category'], 'district': 'Kathmandu', 'time_slot': 'morning', 'hour': 9,
                'day_of_week': random.randint(0, 6), 'month': random.randint(3, 5), 'day': random.randint(1, 28),
                'season': 'Spring', 'is_weekend': 1, 'is_holiday': 0, 'weather_condition': 'Sunny',
                'crowdlevel': random.uniform(75, 95), 'tourist_season': 'Peak', 'festival_period': 'No',
            })
        for _ in range(10):
            synthetic_rows.append({
                'place_id': p['id'], 'category': p['category'], 'district': 'Kathmandu', 'time_slot': 'morning', 'hour': 9,
                'day_of_week': random.randint(0, 6), 'month': random.randint(3, 5), 'day': random.randint(1, 28),
                'season': 'Spring', 'is_weekend': 0, 'is_holiday': 0, 'weather_condition': 'Rainy',
                'crowdlevel': random.uniform(10, 25), 'tourist_season': 'Peak', 'festival_period': 'No',
            })
    else:
        # Others: mostly low
        for _ in range(20):
            synthetic_rows.append({
                'place_id': p['id'], 'category': p['category'], 'district': 'Kathmandu', 'time_slot': 'morning', 'hour': 9,
                'day_of_week': random.randint(0, 6), 'month': random.randint(3, 5), 'day': random.randint(1, 28),
                'season': 'Spring', 'is_weekend': 0, 'is_holiday': 0, 'weather_condition': 'Rainy',
                'crowdlevel': random.uniform(10, 25), 'tourist_season': 'Peak', 'festival_period': 'No',
            })
        for _ in range(5):
            synthetic_rows.append({
                'place_id': p['id'], 'category': p['category'], 'district': 'Kathmandu', 'time_slot': 'morning', 'hour': 9,
                'day_of_week': random.randint(0, 6), 'month': random.randint(3, 5), 'day': random.randint(1, 28),
                'season': 'Spring', 'is_weekend': 1, 'is_holiday': 0, 'weather_condition': 'Sunny',
                'crowdlevel': random.uniform(40, 65), 'tourist_season': 'Peak', 'festival_period': 'No',
            })
        for _ in range(2):
            synthetic_rows.append({
                'place_id': p['id'], 'category': p['category'], 'district': 'Kathmandu', 'time_slot': 'morning', 'hour': 9,
                'day_of_week': random.randint(0, 6), 'month': random.randint(3, 5), 'day': random.randint(1, 28),
                'season': 'Spring', 'is_weekend': 1, 'is_holiday': 0, 'weather_condition': 'Sunny',
                'crowdlevel': random.uniform(75, 95), 'tourist_season': 'Peak', 'festival_period': 'No',
            })
# --- End realistic morning logic ---

# --- Realistic 3-2-2 pattern for Kathmandu, afternoon ---
kathmandu_places_afternoon = [
    # 3 High crowd (tourist/famous)
    {'id': 70, 'name': 'Thamel', 'category': 'Market', 'crowd': 'high'},
    {'id': 130, 'name': 'Kathmandu Durbar Square', 'category': 'Historical', 'crowd': 'high'},
    {'id': 68, 'name': 'Pashupatinath Temple', 'category': 'Temple', 'crowd': 'high'},
    # 2 Medium crowd (parks/hill stations)
    {'id': 69, 'name': 'Garden of Dreams', 'category': 'Nature', 'crowd': 'medium'},
    {'id': 134, 'name': 'Shivapuri National Park', 'category': 'Nature', 'crowd': 'medium'},
    # 2 Low crowd (quiet/less popular)
    {'id': 67, 'name': 'Sankhu', 'category': 'Cultural', 'crowd': 'low'},
    {'id': 83, 'name': 'Sundarijal', 'category': 'Nature', 'crowd': 'low'},
]
for p in kathmandu_places_afternoon:
    if p['crowd'] == 'high':
        for _ in range(25):
            synthetic_rows.append({
                'place_id': p['id'], 'category': p['category'], 'district': 'Kathmandu', 'time_slot': 'afternoon', 'hour': 14,
                'day_of_week': random.randint(0, 6), 'month': random.randint(3, 5), 'day': random.randint(1, 28),
                'season': 'Spring', 'is_weekend': random.choice([0, 1]), 'is_holiday': 0, 'weather_condition': 'Sunny',
                'crowdlevel': random.uniform(75, 95), 'tourist_season': 'Peak', 'festival_period': 'No',
            })
    elif p['crowd'] == 'medium':
        for _ in range(15):
            synthetic_rows.append({
                'place_id': p['id'], 'category': p['category'], 'district': 'Kathmandu', 'time_slot': 'afternoon', 'hour': 14,
                'day_of_week': random.randint(0, 6), 'month': random.randint(3, 5), 'day': random.randint(1, 28),
                'season': 'Spring', 'is_weekend': random.choice([0, 1]), 'is_holiday': 0, 'weather_condition': 'Sunny',
                'crowdlevel': random.uniform(40, 65), 'tourist_season': 'Peak', 'festival_period': 'No',
            })
    else:
        for _ in range(10):
            synthetic_rows.append({
                'place_id': p['id'], 'category': p['category'], 'district': 'Kathmandu', 'time_slot': 'afternoon', 'hour': 14,
                'day_of_week': random.randint(0, 6), 'month': random.randint(3, 5), 'day': random.randint(1, 28),
                'season': 'Spring', 'is_weekend': random.choice([0, 1]), 'is_holiday': 0, 'weather_condition': 'Sunny',
                'crowdlevel': random.uniform(10, 25), 'tourist_season': 'Peak', 'festival_period': 'No',
            })
# --- End 3-2-2 pattern for Kathmandu, afternoon ---

# --- Guaranteed 3-2-2 pattern for Kathmandu, morning and afternoon ---
kathmandu_322_places = [
    # 3 High crowd
    {'id': 1, 'name': 'Pashupatinath Temple', 'category': 'Religious', 'crowd': 'high'},
    {'id': 2, 'name': 'Garden of Dreams', 'category': 'Park', 'crowd': 'high'},
    {'id': 3, 'name': 'Asan Market', 'category': 'Market', 'crowd': 'high'},
    # 2 Medium crowd
    {'id': 4, 'name': 'Swayambhunath', 'category': 'Religious', 'crowd': 'medium'},
    {'id': 5, 'name': 'Bhadrakali Temple', 'category': 'Religious', 'crowd': 'medium'},
    # 2 Low crowd
    {'id': 6, 'name': 'National Library', 'category': 'Museum', 'crowd': 'low'},
    {'id': 7, 'name': 'Chhauni Museum', 'category': 'Museum', 'crowd': 'low'},
]
for time_slot, hour in [('morning', 9), ('afternoon', 14), ('evening', 18)]:
    for p in kathmandu_322_places:
        if p['crowd'] == 'high':
            for _ in range(100):
                synthetic_rows.append({
                    'place_id': p['id'], 'category': p['category'], 'district': 'Kathmandu', 'time_slot': time_slot, 'hour': hour,
                    'day_of_week': 2, 'month': 6, 'day': 15, 'season': 'Summer', 'is_weekend': 1, 'is_holiday': 0, 'weather_condition': 'Sunny',
                    'crowdlevel': random.uniform(80, 90), 'tourist_season': 'Peak', 'festival_period': 'No',
                })
        elif p['crowd'] == 'medium':
            for _ in range(100):
                synthetic_rows.append({
                    'place_id': p['id'], 'category': p['category'], 'district': 'Kathmandu', 'time_slot': time_slot, 'hour': hour,
                    'day_of_week': 2, 'month': 6, 'day': 15, 'season': 'Summer', 'is_weekend': 1, 'is_holiday': 0, 'weather_condition': 'Sunny',
                    'crowdlevel': random.uniform(50, 65), 'tourist_season': 'Peak', 'festival_period': 'No',
                })
        else:
            for _ in range(100):
                synthetic_rows.append({
                    'place_id': p['id'], 'category': p['category'], 'district': 'Kathmandu', 'time_slot': time_slot, 'hour': hour,
                    'day_of_week': 2, 'month': 6, 'day': 15, 'season': 'Summer', 'is_weekend': 1, 'is_holiday': 0, 'weather_condition': 'Sunny',
                    'crowdlevel': random.uniform(15, 35), 'tourist_season': 'Peak', 'festival_period': 'No',
                })
print('✅ Guaranteed 3-2-2 training pattern injected for Kathmandu, morning, afternoon, and evening.')
# --- End guaranteed 3-2-2 pattern ---

for _ in range(2200):
    cat, district = random.choice(categories)
    time_slot = random.choice(time_slots)
    weather = random.choice(weather_options)
    season = random.choice(seasons)
    month = random.randint(1, 12)
    day_of_week = random.randint(0, 6)
    is_weekend = 1 if day_of_week >= 5 else 0
    is_holiday = random.choices([0, 1], weights=[0.85, 0.15])[0]
    hour = 9 if time_slot == 'morning' else 14 if time_slot == 'afternoon' else 19
    place_id = random.randint(10000, 20000)
    # Nepali crowd rules (more balanced)
    if cat == 'Religious':
        if time_slot == 'morning':
            base = random.uniform(60, 85)
        elif time_slot == 'evening':
            base = random.uniform(45, 65)
        else:
            base = random.uniform(35, 55)
    elif cat == 'Park':
        if time_slot == 'morning':
            base = random.uniform(50, 75)
        elif time_slot == 'evening':
            base = random.uniform(60, 80)
        else:
            base = random.uniform(25, 45)
    elif cat == 'Market':
        if time_slot == 'evening':
            base = random.uniform(70, 90)
        elif time_slot == 'morning':
            base = random.uniform(20, 40)
        else:
            base = random.uniform(45, 65)
    elif cat in ['Tourist', 'Museum']:
        if time_slot == 'afternoon':
            base = random.uniform(60, 80)
        else:
            base = random.uniform(25, 45)
    else:
        base = random.uniform(25, 65)
    # Weather effect (less aggressive)
    if weather == 'Sunny':
        base *= 1.02  # Only 2% increase
    elif weather == 'Rainy':
        base *= 0.95  # Only 5% decrease
    elif weather == 'Cloudy':
        base *= 0.98  # Slight decrease for cloudy
    
    # Add some variation but keep it reasonable
    variation = random.uniform(-3, 3)
    crowdlevel = max(10, min(100, round(base + variation, 1)))  # Minimum 10% crowd
    row = {
        'place_id': place_id,
        'category': cat,
        'district': district,
        'time_slot': time_slot,
        'hour': hour,
        'day_of_week': day_of_week,
        'month': month,
        'day': random.randint(1, 28),
        'season': season,
        'is_weekend': is_weekend,
        'is_holiday': is_holiday,
        'weather_condition': weather,
        'crowdlevel': crowdlevel,
        'tourist_season': 'Peak' if month in [10, 11, 4, 5] else 'Shoulder' if month in [3, 9, 12] else 'Low',
        'festival_period': 'Yes' if cat == 'Religious' and random.random() < 0.1 else 'No'
    }
    synthetic_rows.append(row)

synth_df = pd.DataFrame(synthetic_rows)

# 3. Combine and save
enhanced_df = pd.concat([df, synth_df], ignore_index=True)
enhanced_csv = 'enhanced_crowd_data.csv'
enhanced_df.to_csv(enhanced_csv, index=False)

# 4. Train model
model = ImprovedCrowdPredictionModel('crowd_prediction_model.joblib')
model.train_model(enhanced_csv)

# 5. Log metrics
info = model.get_model_info()
print(f"\nMean Squared Error: {info.get('mse', 0):.2f}")
print(f"R² Score: {info.get('r2_score', 0):.4f}")

# 6. Feature importance (top 5)
if hasattr(model.model, 'feature_importances_'):
    importances = model.model.feature_importances_
    features = model.feature_names
    fi_df = pd.DataFrame({'feature': features, 'importance': importances})
    fi_df = fi_df.sort_values('importance', ascending=False)
    print("\nTop 5 Feature Importances:")
    print(fi_df.head(5))

# 7. Test predictions
print("\nTest predictions:")
test_cases = [
    {'name': 'Park, Saturday morning, sunny', 'params': {'place_id': 12345, 'category': 'Park', 'district': 'Lalitpur', 'time_slot': 'morning', 'day_of_week': 5, 'month': 4, 'season': 'Spring', 'is_weekend': 1, 'is_holiday': 0, 'weather_condition': 'Sunny', 'hour': 9}},
    {'name': 'Temple, Sunday morning, rainy', 'params': {'place_id': 12346, 'category': 'Religious', 'district': 'Kathmandu', 'time_slot': 'morning', 'day_of_week': 6, 'month': 11, 'season': 'Autumn', 'is_weekend': 1, 'is_holiday': 0, 'weather_condition': 'Rainy', 'hour': 9}},
    {'name': 'Museum, Wednesday afternoon, cloudy', 'params': {'place_id': 12347, 'category': 'Museum', 'district': 'Kathmandu', 'time_slot': 'afternoon', 'day_of_week': 2, 'month': 7, 'season': 'Summer', 'is_weekend': 0, 'is_holiday': 0, 'weather_condition': 'Cloudy', 'hour': 14}},
]
for case in test_cases:
    pred = model.predict(**case['params'])
    print(f"{case['name']}: {pred:.1f}% crowd level")

print("\n✅ Model trained with real + smart Nepali logic!") 