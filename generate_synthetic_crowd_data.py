import csv
import os
import django
import random
from datetime import datetime

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'mainfolder.settings')
django.setup()

from backend.models import Place

# Settings
OUTPUT_CSV = 'enhanced_crowd_training_data.csv'
TIME_SLOTS = ['morning', 'afternoon', 'evening']
SEASONS = ['Winter', 'Spring', 'Summer', 'Fall']
WEATHER = ['Sunny', 'Rainy', 'Cloudy', 'Foggy']
DAYS = list(range(7))  # 0=Monday, 6=Sunday
MONTHS = list(range(1, 13))

# Helper: realistic crowdlevel generator
CATEGORY_BASE = {
    'Temple': {'morning': (60, 100), 'afternoon': (20, 50), 'evening': (10, 40)},
    'Park': {'morning': (30, 60), 'afternoon': (40, 80), 'evening': (60, 100)},
    'Market': {'morning': (20, 50), 'afternoon': (50, 100), 'evening': (30, 70)},
    'Historical': {'morning': (10, 40), 'afternoon': (30, 70), 'evening': (20, 60)},
    'Travel': {'morning': (10, 40), 'afternoon': (20, 60), 'evening': (20, 60)},
}

def get_crowdlevel(category, time_slot, is_weekend, is_holiday, weather):
    base = CATEGORY_BASE.get(category, CATEGORY_BASE['Travel'])
    low, high = base[time_slot]
    crowd = random.randint(low, high)
    # Boost for weekend/holiday
    if is_weekend or is_holiday:
        crowd = int(crowd * random.uniform(1.1, 1.3))
    # Reduce for bad weather
    if weather in ['Rainy', 'Foggy']:
        crowd = int(crowd * random.uniform(0.6, 0.85))
    return max(0, min(100, crowd))

def main():
    places = Place.objects.all()
    with open(OUTPUT_CSV, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['place_id','category','district','time_slot','hour','day_of_week','month','season','is_weekend','is_holiday','weather_condition','crowdlevel'])
        for place in places:
            for time_slot in TIME_SLOTS:
                for day_of_week in DAYS:
                    for month in MONTHS:
                        season = SEASONS[((month-1)//3)%4]
                        is_weekend = 1 if day_of_week in [5,6] else 0
                        for is_holiday in [0, 1]:
                            for weather in WEATHER:
                                hour = {'morning': 8, 'afternoon': 14, 'evening': 19}[time_slot]
                                crowdlevel = get_crowdlevel(place.category, time_slot, is_weekend, is_holiday, weather)
                                writer.writerow([
                                    place.id,
                                    place.category,
                                    place.district,
                                    time_slot,
                                    hour,
                                    day_of_week,
                                    month,
                                    season,
                                    is_weekend,
                                    is_holiday,
                                    weather,
                                    crowdlevel
                                ])
    print(f"Synthetic crowd data written to {OUTPUT_CSV}")

if __name__ == '__main__':
    main() 