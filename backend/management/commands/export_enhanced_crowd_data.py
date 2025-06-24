from django.core.management.base import BaseCommand
import csv
from backend.models import Place
from datetime import datetime, timedelta
import random
import numpy as np

def get_hour_for_timeslot(time_slot):
    """Get a random hour within the time slot"""
    if time_slot == 'morning':
        return random.randint(6, 11)
    elif time_slot == 'afternoon':
        return random.randint(12, 16)
    else:  # evening
        return random.randint(17, 20)

def get_base_crowd(category, tags, time_slot, previous_crowd=None):
    """Get base crowd level with more realistic patterns"""
    # Base crowd by category
    category_base = {
        'Temple': 60,
        'Restaurant': 50,
        'Park': 45,
        'Museum': 40,
        'Shopping': 55,
        'Entertainment': 65,
        'Cultural': 50,
        'Historical': 45,
        'Nature': 40,
    }.get(category, 50)
    
    # Time slot adjustments
    time_adjustments = {
        'morning': lambda x: x * 0.8,  # Less crowded in morning
        'afternoon': lambda x: x * 1.2,  # Peak time
        'evening': lambda x: x * 1.1,  # Slightly less than afternoon
    }
    
    base = category_base
    base = time_adjustments[time_slot](base)
    
    # Tag-based adjustments
    tag_effects = {
        'popular': 15,
        'tourist': 10,
        'local': -5,
        'quiet': -10,
        'busy': 15,
        'peaceful': -15,
    }
    
    for tag in tags:
        tag = tag.lower()
        if tag in tag_effects:
            base += tag_effects[tag]
    
    # If we have previous crowd data, make changes more gradual
    if previous_crowd is not None:
        # Maximum allowed change in 30 minutes (as percentage points)
        max_change = 15
        difference = base - previous_crowd
        if abs(difference) > max_change:
            # Make the change more gradual
            base = previous_crowd + (max_change if difference > 0 else -max_change)
    
    return max(0, min(100, base))

class Command(BaseCommand):
    help = 'Export enhanced crowd data for model training with realistic patterns'

    def handle(self, *args, **options):
        filename = 'enhanced_crowd_training_data.csv'
        self.stdout.write(f'Exporting crowd data to {filename}...')
        
        places = Place.objects.all()
        if not places.exists():
            self.stdout.write(self.style.ERROR('No places found in the database!'))
            return
        
        # Simulate for 1 year
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([
                'place_id', 'category', 'district', 'time_slot', 'hour', 'day_of_week', 
                'month', 'season', 'is_weekend', 'is_holiday', 'weather_condition', 'crowdlevel'
            ])
            
            # Weather patterns by season
            weather_by_season = {
                'Winter': (['Sunny', 'Cloudy', 'Foggy'], [0.3, 0.4, 0.3]),
                'Spring': (['Sunny', 'Cloudy', 'Rainy'], [0.4, 0.3, 0.3]),
                'Summer': (['Sunny', 'Cloudy', 'Rainy'], [0.5, 0.3, 0.2]),
                'Autumn': (['Sunny', 'Cloudy', 'Rainy', 'Foggy'], [0.3, 0.3, 0.2, 0.2])
            }
            
            # Holidays (major festivals and events)
            holidays = set([
                '2025-01-01',  # New Year
                '2025-04-14',  # Nepali New Year
                '2025-05-01',  # Labor Day
                '2025-08-19',  # Janai Purnima
                '2025-09-19',  # Constitution Day
                '2025-10-20',  # Dashain
                '2025-11-09',  # Tihar
                '2025-12-25',  # Christmas
            ])
            
            # For each place, simulate for every day and time slot
            for place in places:
                tags = list(place.tags.values_list('name', flat=True)) if hasattr(place, 'tags') else []
                current_date = start_date
                previous_crowd = None
                
                for _ in range(366):
                    # Determine season
                    month = current_date.month
                    if month in [12, 1, 2]:
                        season = 'Winter'
                    elif month in [3, 4, 5]:
                        season = 'Spring'
                    elif month in [6, 7, 8]:
                        season = 'Summer'
                    else:
                        season = 'Autumn'
                    
                    # Get weather conditions for this season
                    conditions, weights = weather_by_season[season]
                    weather = np.random.choice(conditions, p=weights)
                    
                    for time_slot in ['morning', 'afternoon', 'evening']:
                        # Get random hour for this time slot
                        hour = get_hour_for_timeslot(time_slot)
                        
                        # Get base crowd with previous crowd for gradual changes
                        base = get_base_crowd(place.category, tags, time_slot, previous_crowd)
                        
                        # Weekend effect
                        is_weekend = 1 if current_date.weekday() >= 5 else 0
                        if is_weekend:
                            base += random.uniform(10, 20)
                        
                        # Holiday effect
                        is_holiday = 1 if current_date.strftime('%Y-%m-%d') in holidays else 0
                        if is_holiday:
                            base += random.uniform(15, 30)
                        
                        # Weather effects
                        weather_effects = {
                            'Rainy': lambda x: x * 0.7,  # 30% reduction
                            'Foggy': lambda x: x * 0.8,  # 20% reduction
                            'Cloudy': lambda x: x * 0.9,  # 10% reduction
                            'Sunny': lambda x: x * 1.1,   # 10% increase
                        }
                        base = weather_effects.get(weather, lambda x: x)(base)
                        
                        # Add small random variation (±5%)
                        variation = random.uniform(-5, 5)
                        crowd_level = max(0, min(100, base + variation))
                        
                        # Store for next iteration
                        previous_crowd = crowd_level
                        
                        writer.writerow([
                            place.id,
                            place.category,
                            place.district,
                            time_slot,
                            hour,
                            current_date.weekday(),
                            current_date.month,
                            season,
                            is_weekend,
                            is_holiday,
                            weather,
                            round(crowd_level, 1)
                        ])
                    
                    current_date += timedelta(days=1)
                
        self.stdout.write(self.style.SUCCESS('Successfully exported enhanced crowd data!')) 