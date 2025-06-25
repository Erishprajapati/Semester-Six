from django.core.management.base import BaseCommand
from backend.models import Place
import csv

class Command(BaseCommand):
    help = 'Generate a logical crowd pattern CSV for all places and tags in the database.'

    def handle(self, *args, **options):
        CATEGORY_PATTERNS = {
            'temple':    [85, 90, 80, 60, 40, 30, 20, 20, 20, 20, 20, 20, 20, 20, 20, 30, 40, 50, 60, 70, 80, 90, 95, 90],
            'park':      [80, 85, 80, 60, 40, 30, 20, 20, 20, 20, 20, 20, 20, 20, 20, 30, 40, 50, 60, 70, 80, 85, 80, 75],
            'market':    [20, 20, 20, 20, 20, 20, 30, 40, 50, 60, 70, 80, 90, 95, 90, 80, 70, 60, 50, 40, 30, 20, 20, 20],
            'tourist':   [30, 30, 30, 30, 30, 30, 40, 50, 60, 70, 80, 90, 90, 90, 90, 80, 70, 60, 50, 40, 30, 30, 30, 30],
            'heritage':  [40, 40, 40, 40, 40, 40, 50, 60, 70, 80, 90, 90, 90, 90, 90, 80, 70, 60, 50, 40, 40, 40, 40, 40],
            'nature':    [60, 65, 60, 50, 40, 30, 20, 20, 20, 20, 20, 20, 20, 20, 20, 30, 40, 50, 60, 70, 75, 80, 80, 70],
        }
        DEFAULT_PATTERN = [30] * 24
        csv_file = 'enhanced_crowd_training_data.csv'
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['place_id','category','district','time_slot','hour','day_of_week','month','season','is_weekend','is_holiday','weather_condition','crowdlevel'])
            for place in Place.objects.all():
                category = (place.category or '').lower()
                pattern = CATEGORY_PATTERNS.get(category, DEFAULT_PATTERN)
                for hour in range(24):
                    if 5 <= hour < 12:
                        time_slot = 'morning'
                    elif 12 <= hour < 17:
                        time_slot = 'afternoon'
                    elif 17 <= hour < 21:
                        time_slot = 'evening'
                    else:
                        time_slot = 'morning'
                    for day_of_week in range(7):
                        for month in range(1, 13):
                            season = 'Spring' if month in [3,4,5] else 'Summer' if month in [6,7,8] else 'Fall' if month in [9,10,11] else 'Winter'
                            is_weekend = 1 if day_of_week >= 5 else 0
                            is_holiday = 0
                            weather_condition = 'Sunny'
                            crowdlevel = pattern[hour]
                            if category in ['tourist', 'heritage'] and is_weekend:
                                crowdlevel = min(100, crowdlevel + 20)
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
                                weather_condition,
                                crowdlevel
                            ])
        self.stdout.write(self.style.SUCCESS(f'Logical crowd pattern CSV generated for all places: {csv_file}')) 