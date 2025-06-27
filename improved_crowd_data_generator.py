import csv
import os
import django
import random
import numpy as np
from datetime import datetime, timedelta

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'mainfolder.settings')
django.setup()

from backend.models import Place

class ImprovedCrowdDataGenerator:
    def __init__(self):
        self.output_file = 'improved_crowd_training_data.csv'
        
        # More realistic base crowd levels by category and time
        self.category_base_levels = {
            'Temple': {
                'morning': (40, 80),    # High in morning for prayers
                'afternoon': (20, 50),  # Moderate in afternoon
                'evening': (30, 70)     # Moderate-high in evening
            },
            'Historical': {
                'morning': (20, 50),    # Moderate in morning
                'afternoon': (40, 80),  # High in afternoon (tourist time)
                'evening': (15, 40)     # Low in evening
            },
            'Market': {
                'morning': (30, 60),    # Moderate in morning
                'afternoon': (60, 95),  # Very high in afternoon
                'evening': (40, 75)     # High in evening
            },
            'Park': {
                'morning': (20, 45),    # Low-moderate in morning
                'afternoon': (30, 65),  # Moderate in afternoon
                'evening': (50, 85)     # High in evening (exercise time)
            },
            'Cultural': {
                'morning': (15, 35),    # Low in morning
                'afternoon': (25, 55),  # Moderate in afternoon
                'evening': (35, 70)     # Moderate-high in evening
            },
            'Nature': {
                'morning': (25, 50),    # Moderate in morning
                'afternoon': (35, 65),  # Moderate-high in afternoon
                'evening': (20, 45)     # Moderate in evening
            },
            'Religious': {
                'morning': (35, 75),    # High in morning
                'afternoon': (25, 55),  # Moderate in afternoon
                'evening': (30, 65)     # Moderate in evening
            }
        }
        
        # Weather effects (multipliers)
        self.weather_effects = {
            'Sunny': 1.1,      # 10% increase
            'Cloudy': 0.95,    # 5% decrease
            'Rainy': 0.6,      # 40% decrease
            'Foggy': 0.7       # 30% decrease
        }
        
        # Season definitions (corrected)
        self.season_months = {
            'Winter': [12, 1, 2],
            'Spring': [3, 4, 5],
            'Summer': [6, 7, 8],
            'Autumn': [9, 10, 11]
        }
        
        # Weather patterns by season
        self.season_weather = {
            'Winter': {'Sunny': 0.3, 'Cloudy': 0.4, 'Foggy': 0.3},
            'Spring': {'Sunny': 0.4, 'Cloudy': 0.3, 'Rainy': 0.3},
            'Summer': {'Sunny': 0.5, 'Cloudy': 0.3, 'Rainy': 0.2},
            'Autumn': {'Sunny': 0.3, 'Cloudy': 0.3, 'Rainy': 0.2, 'Foggy': 0.2}
        }
        
        # Major holidays in Nepal
        self.holidays = [
            '2025-01-01',  # New Year
            '2025-04-14',  # Nepali New Year
            '2025-05-01',  # Labor Day
            '2025-08-19',  # Janai Purnima
            '2025-09-19',  # Constitution Day
            '2025-10-20',  # Dashain
            '2025-11-09',  # Tihar
            '2025-12-25',  # Christmas
        ]
    
    def get_season(self, month):
        """Get season based on month"""
        for season, months in self.season_months.items():
            if month in months:
                return season
        return 'Spring'  # Default fallback
    
    def get_weather(self, season):
        """Get weather condition based on season probabilities"""
        weather_probs = self.season_weather[season]
        weather_conditions = list(weather_probs.keys())
        probabilities = list(weather_probs.values())
        return np.random.choice(weather_conditions, p=probabilities)
    
    def get_hour(self, time_slot):
        """Get realistic hour for time slot"""
        if time_slot == 'morning':
            return random.randint(7, 11)
        elif time_slot == 'afternoon':
            return random.randint(12, 17)
        else:  # evening
            return random.randint(18, 21)
    
    def calculate_crowd_level(self, place, time_slot, day_of_week, month, 
                            is_weekend, is_holiday, weather, previous_crowd=None):
        """Calculate realistic crowd level with gradual changes"""
        
        # Get base crowd range for category and time slot
        category = place.category
        base_range = self.category_base_levels.get(category, self.category_base_levels['Cultural'])
        low, high = base_range[time_slot]
        
        # Start with base crowd level
        crowd_level = random.uniform(low, high)
        
        # Weekend effect (20-40% increase)
        if is_weekend:
            crowd_level *= random.uniform(1.2, 1.4)
        
        # Holiday effect (30-60% increase)
        if is_holiday:
            crowd_level *= random.uniform(1.3, 1.6)
        
        # Weather effect
        weather_multiplier = self.weather_effects.get(weather, 1.0)
        crowd_level *= weather_multiplier
        
        # Day of week effect
        if day_of_week == 0:  # Monday - lower
            crowd_level *= 0.9
        elif day_of_week == 4:  # Friday - higher
            crowd_level *= 1.1
        elif day_of_week == 5:  # Saturday - higher
            crowd_level *= 1.15
        elif day_of_week == 6:  # Sunday - moderate
            crowd_level *= 1.05
        
        # Month/season effect
        if month in [6, 7, 8]:  # Summer - tourist season
            crowd_level *= 1.1
        elif month in [12, 1, 2]:  # Winter - less tourists
            crowd_level *= 0.9
        
        # Gradual changes (if previous crowd data exists)
        if previous_crowd is not None:
            max_change = 15  # Maximum 15% change
            difference = crowd_level - previous_crowd
            if abs(difference) > max_change:
                if difference > 0:
                    crowd_level = previous_crowd + max_change
                else:
                    crowd_level = previous_crowd - max_change
        
        # Add small random variation (±5%)
        variation = random.uniform(-5, 5)
        crowd_level += variation
        
        # Ensure crowd level is within bounds
        return max(5, min(95, crowd_level))
    
    def generate_data(self):
        """Generate improved crowd training data"""
        print("Starting improved crowd data generation...")
        
        places = Place.objects.all()
        if not places.exists():
            print("No places found in database!")
            return
        
        # Generate data for 6 months (more manageable dataset)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=180)
        
        with open(self.output_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([
                'place_id', 'category', 'district', 'time_slot', 'hour', 
                'day_of_week', 'month', 'season', 'is_weekend', 'is_holiday', 
                'weather_condition', 'crowdlevel'
            ])
            
            total_rows = 0
            
            for place in places:
                print(f"Processing place: {place.name} ({place.category})")
                current_date = start_date
                previous_crowd = None
                
                for day in range(180):
                    # Determine season
                    season = self.get_season(current_date.month)
                    
                    # Check if it's a holiday
                    is_holiday = current_date.strftime('%Y-%m-%d') in self.holidays
                    
                    # Check if it's weekend
                    is_weekend = current_date.weekday() >= 5
                    
                    for time_slot in ['morning', 'afternoon', 'evening']:
                        # Get weather for this season
                        weather = self.get_weather(season)
                        
                        # Get hour for time slot
                        hour = self.get_hour(time_slot)
                        
                        # Calculate crowd level
                        crowd_level = self.calculate_crowd_level(
                            place, time_slot, current_date.weekday(), 
                            current_date.month, is_weekend, is_holiday, 
                            weather, previous_crowd
                        )
                        
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
                            1 if is_weekend else 0,
                            1 if is_holiday else 0,
                            weather,
                            round(crowd_level, 1)
                        ])
                        
                        total_rows += 1
                    
                    current_date += timedelta(days=1)
        
        print(f"Generated {total_rows} rows of improved crowd data")
        print(f"Data saved to: {self.output_file}")
        
        # Validate the generated data
        self.validate_data()
    
    def validate_data(self):
        """Validate the generated data for quality"""
        print("\n=== VALIDATING GENERATED DATA ===")
        
        import pandas as pd
        df = pd.read_csv(self.output_file)
        
        print(f"Total rows: {len(df)}")
        print(f"Unique places: {df['place_id'].nunique()}")
        print(f"Date range: {df['month'].min()} - {df['month'].max()}")
        
        # Check crowd level distribution
        print(f"\nCrowd level statistics:")
        print(df['crowdlevel'].describe())
        
        # Check by category
        print(f"\nCrowd levels by category:")
        print(df.groupby('category')['crowdlevel'].agg(['mean', 'std', 'min', 'max']))
        
        # Check by time slot
        print(f"\nCrowd levels by time slot:")
        print(df.groupby('time_slot')['crowdlevel'].agg(['mean', 'std', 'min', 'max']))
        
        # Check season consistency
        print(f"\nSeason distribution:")
        print(df.groupby(['month', 'season']).size().unstack(fill_value=0))
        
        # Check for extreme values
        extreme_low = (df['crowdlevel'] < 5).sum()
        extreme_high = (df['crowdlevel'] > 95).sum()
        print(f"\nExtreme values:")
        print(f"  Very low (< 5): {extreme_low}")
        print(f"  Very high (> 95): {extreme_high}")
        
        print("\nData validation completed!")

if __name__ == '__main__':
    generator = ImprovedCrowdDataGenerator()
    generator.generate_data() 