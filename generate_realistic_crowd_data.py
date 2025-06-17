import csv
import random
from datetime import datetime, timedelta
import django
import os
import sys

# Setup Django environment
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'mainfolder.settings')
django.setup()

# Now import Django models after setup
from backend.models import Place, Tag

# Helper functions
SEASONS = {1: 'Winter', 2: 'Winter', 3: 'Spring', 4: 'Spring', 5: 'Spring', 6: 'Summer', 7: 'Summer', 8: 'Summer', 9: 'Fall', 10: 'Fall', 11: 'Fall', 12: 'Winter'}
WEATHER = ['Sunny', 'Cloudy', 'Rainy', 'Clear']

# Simulate some holidays (Nepali and global)
HOLIDAYS = [(1, 1), (2, 19), (5, 1), (9, 20), (10, 2), (11, 11)]

# Place type patterns
PATTERNS = {
    'Temple': {'morning': (60, 20), 'afternoon': (40, 15), 'evening': (50, 15)},
    'Park': {'morning': (20, 10), 'afternoon': (40, 15), 'evening': (60, 20)},
    'Market': {'morning': (30, 10), 'afternoon': (60, 20), 'evening': (80, 20)},
    'Museum': {'morning': (25, 10), 'afternoon': (50, 15), 'evening': (35, 10)},
    'Stupa': {'morning': (50, 15), 'afternoon': (35, 10), 'evening': (45, 10)},
    'Palace': {'morning': (25, 10), 'afternoon': (45, 15), 'evening': (35, 10)},
    'Garden': {'morning': (15, 10), 'afternoon': (35, 10), 'evening': (50, 15)},
    'Viewpoint': {'morning': (30, 10), 'afternoon': (45, 10), 'evening': (55, 15)},
    'Nature': {'morning': (25, 10), 'afternoon': (35, 10), 'evening': (30, 10)},
    'Cultural': {'morning': (35, 10), 'afternoon': (50, 15), 'evening': (45, 10)},
}

# Get all places
places = Place.objects.all()

# Output file
output_file = 'realistic_crowd_training_data.csv'

with open(output_file, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow([
        'place_id', 'place_name', 'category', 'district', 'tags',
        'date', 'time_slot', 'day_of_week', 'month', 'season', 'is_weekend',
        'is_holiday', 'weather_condition', 'crowdlevel', 'status'
    ])
    
    start_date = datetime.now() - timedelta(days=180)
    end_date = datetime.now()
    
    for place in places:
        tags = ','.join([tag.name for tag in place.tags.all()])
        category = place.category
        for i in range(181):
            date = start_date + timedelta(days=i)
            day_of_week = date.weekday()
            month = date.month
            season = SEASONS[month]
            is_weekend = 1 if day_of_week >= 5 else 0
            is_holiday = 1 if (month, date.day) in HOLIDAYS else 0
            weather = random.choices(WEATHER, weights=[0.5, 0.2, 0.2, 0.1])[0]
            for time_slot in ['morning', 'afternoon', 'evening']:
                # Pick pattern based on category or tags
                pattern = PATTERNS.get(category, PATTERNS.get('Temple'))
                mean, std = pattern.get(time_slot, (40, 15))
                # Add effects
                crowd = random.gauss(mean, std)
                if is_weekend:
                    crowd += random.randint(5, 20)
                if is_holiday:
                    crowd += random.randint(10, 30)
                if weather == 'Rainy':
                    crowd -= random.randint(5, 20)
                # Add some missing values
                if random.random() < 0.01:
                    continue  # skip this row
                # Add outliers
                if random.random() < 0.01:
                    crowd += random.randint(30, 50)
                # Clamp
                crowd = max(0, min(100, int(crowd)))
                status = 'High' if crowd > 70 else 'Medium' if crowd > 30 else 'Low'
                writer.writerow([
                    place.id, place.name, category, place.district, tags,
                    date.strftime('%Y-%m-%d'), time_slot, day_of_week, month, season, is_weekend,
                    is_holiday, weather, crowd, status
                ])
print(f"Generated {output_file} with realistic, varied crowd data!") 