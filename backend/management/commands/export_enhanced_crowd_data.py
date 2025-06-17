import csv
from django.core.management.base import BaseCommand
from backend.models import CrowdData, Place
from datetime import datetime, timedelta
import random

class Command(BaseCommand):
    help = 'Export enhanced crowd data for machine learning training'

    def handle(self, *args, **kwargs):
        self.stdout.write('Generating enhanced training dataset...')
        
        # Create enhanced dataset with more features
        with open('enhanced_crowd_training_data.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                'place_id', 'place_name', 'category', 'district', 'tags',
                'time_slot', 'day_of_week', 'month', 'season', 'is_weekend',
                'is_holiday', 'weather_condition', 'crowdlevel', 'status'
            ])
            
            # Generate 6 months of historical data
            start_date = datetime.now() - timedelta(days=180)
            end_date = datetime.now()
            
            places = Place.objects.all()
            
            for place in places:
                current_date = start_date
                while current_date <= end_date:
                    for time_slot in ['morning', 'afternoon', 'evening']:
                        # Generate realistic crowd levels based on patterns
                        base_crowd = self.get_base_crowd_level(place, time_slot)
                        crowd_level = self.apply_patterns(base_crowd, current_date, time_slot)
                        
                        # Determine status
                        if crowd_level > 70:
                            status = 'High'
                        elif crowd_level > 30:
                            status = 'Medium'
                        else:
                            status = 'Low'
                        
                        # Get place features
                        tags = ','.join([tag.name for tag in place.tags.all()])
                        day_of_week = current_date.weekday()
                        month = current_date.month
                        season = self.get_season(month)
                        is_weekend = 1 if day_of_week >= 5 else 0
                        is_holiday = self.is_holiday(current_date)
                        weather = self.get_weather_condition(current_date, month)
                        
                        writer.writerow([
                            place.id, place.name, place.category, place.district, tags,
                            time_slot, day_of_week, month, season, is_weekend,
                            is_holiday, weather, crowd_level, status
                        ])
                    
                    current_date += timedelta(days=1)
        
        self.stdout.write(
            self.style.SUCCESS('Enhanced training dataset created: enhanced_crowd_training_data.csv')
        )
    
    def get_base_crowd_level(self, place, time_slot):
        """Get base crowd level based on place type and time slot"""
        base_levels = {
            'Temple': {'morning': 60, 'afternoon': 40, 'evening': 50},
            'Stupa': {'morning': 50, 'afternoon': 35, 'evening': 45},
            'Museum': {'morning': 30, 'afternoon': 50, 'evening': 40},
            'Palace': {'morning': 25, 'afternoon': 45, 'evening': 35},
            'Park': {'morning': 20, 'afternoon': 40, 'evening': 60},
            'Garden': {'morning': 15, 'afternoon': 35, 'evening': 50},
            'Market': {'morning': 40, 'afternoon': 70, 'evening': 80},
            'Viewpoint': {'morning': 30, 'afternoon': 45, 'evening': 55},
            'Nature': {'morning': 25, 'afternoon': 35, 'evening': 30},
            'Cultural': {'morning': 35, 'afternoon': 50, 'evening': 45},
        }
        
        # Find matching category
        for category, levels in base_levels.items():
            if category.lower() in place.category.lower():
                return levels.get(time_slot, 40)
        
        # Default based on tags
        for tag in place.tags.all():
            if tag.name in base_levels:
                return base_levels[tag.name].get(time_slot, 40)
        
        return 40  # Default
    
    def apply_patterns(self, base_level, date, time_slot):
        """Apply various patterns to base crowd level"""
        crowd_level = base_level
        
        # Weekend effect
        if date.weekday() >= 5:  # Weekend
            crowd_level += random.randint(10, 25)
        
        # Seasonal effect
        month = date.month
        if month in [10, 11, 12, 1, 2]:  # Peak tourist season
            crowd_level += random.randint(5, 15)
        elif month in [6, 7, 8]:  # Monsoon season
            crowd_level -= random.randint(5, 15)
        
        # Time slot specific patterns
        if time_slot == 'morning':
            crowd_level += random.randint(-5, 10)
        elif time_slot == 'afternoon':
            crowd_level += random.randint(0, 15)
        else:  # evening
            crowd_level += random.randint(5, 20)
        
        # Add some randomness
        crowd_level += random.randint(-10, 10)
        
        # Ensure within bounds
        return max(0, min(100, crowd_level))
    
    def get_season(self, month):
        """Get season based on month"""
        if month in [3, 4, 5]:
            return 'Spring'
        elif month in [6, 7, 8]:
            return 'Summer'
        elif month in [9, 10, 11]:
            return 'Fall'
        else:
            return 'Winter'
    
    def is_holiday(self, date):
        """Check if date is a holiday"""
        # Add major Nepali holidays
        holidays = [
            (1, 1),   # New Year
            (2, 19),  # Democracy Day
            (5, 1),   # Labor Day
            (9, 20),  # Constitution Day
            (10, 2),  # Gandhi Jayanti
            (11, 11), # Republic Day
        ]
        
        return 1 if (date.month, date.day) in holidays else 0
    
    def get_weather_condition(self, date, month):
        """Get weather condition based on date"""
        conditions = ['Sunny', 'Cloudy', 'Rainy', 'Clear']
        
        # Monsoon season (June-September)
        if month in [6, 7, 8, 9]:
            return random.choices(conditions, weights=[0.2, 0.3, 0.4, 0.1])[0]
        else:
            return random.choices(conditions, weights=[0.4, 0.3, 0.1, 0.2])[0] 