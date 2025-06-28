#!/usr/bin/env python
"""
Nepal Tourism-Based Crowd Data Generator
Creates realistic crowd data based on actual Nepal tourism patterns and statistics
"""
import os
import sys
import django
import csv
import random
from datetime import datetime, timedelta
import numpy as np

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'mainfolder.settings')
django.setup()

from backend.models import Place

class NepalTourismBasedGenerator:
    def __init__(self, output_file='nepal_tourism_crowd_data.csv'):
        self.output_file = output_file
        
        # Nepal Tourism Statistics 2010-2020 patterns
        self.tourism_patterns = {
            # Peak tourist seasons (October-November, March-May)
            'peak_season_multiplier': 1.8,
            'shoulder_season_multiplier': 1.2,
            'low_season_multiplier': 0.6,
            
            # Monthly tourist arrival patterns (based on Nepal Tourism Board data)
            'monthly_patterns': {
                1: 0.4,   # January - Low season
                2: 0.5,   # February - Low season
                3: 0.8,   # March - Shoulder season
                4: 1.0,   # April - Peak season
                5: 0.9,   # May - Shoulder season
                6: 0.3,   # June - Monsoon (very low)
                7: 0.2,   # July - Monsoon (very low)
                8: 0.3,   # August - Monsoon (very low)
                9: 0.6,   # September - Shoulder season
                10: 1.2,  # October - Peak season
                11: 1.3,  # November - Peak season
                12: 0.7   # December - Shoulder season
            },
            
            # Day of week patterns (weekends are busier)
            'weekend_multiplier': 1.4,
            'weekday_multiplier': 0.8,
            
            # Time slot patterns
            'time_slot_patterns': {
                'morning': 0.7,    # Early morning less crowded
                'afternoon': 1.2,  # Peak visiting hours
                'evening': 0.9     # Moderate evening crowds
            },
            
            # Weather impact on tourism
            'weather_multipliers': {
                'Sunny': 1.1,
                'Cloudy': 1.0,
                'Rainy': 0.6,
                'Foggy': 0.8
            },
            
            # Category-specific base crowd levels (based on tourism preferences)
            'category_base_levels': {
                'Religious': 75,      # High - Pashupatinath, Swayambhunath
                'Historical': 70,     # High - Durbar Squares
                'Market': 85,         # Very High - Thamel, Asan
                'Park': 60,           # Medium - Gardens, Parks
                'Nature': 55,         # Medium - Hills, Viewpoints
                'Natural': 50,        # Medium - Natural sites
                'Cultural': 65,       # High - Cultural sites
                'Museum': 45,         # Medium - Museums
                'Viewpoint': 65,      # High - Scenic viewpoints
                'Monument': 55        # Medium - Monuments
            },
            
            # District-specific multipliers (Kathmandu Valley tourism patterns)
            'district_multipliers': {
                'Kathmandu': 1.2,     # Capital city - highest tourism
                'Lalitpur': 1.0,      # Patan - moderate tourism
                'Bhaktapur': 0.9      # Bhaktapur - moderate tourism
            },
            
            # Special events and festivals (Nepal's major festivals)
            'festival_dates': {
                'Dashain': [(9, 15), (10, 15)],  # September-October
                'Tihar': [(10, 20), (11, 5)],    # October-November
                'Buddha Jayanti': [(4, 25), (5, 15)],  # April-May
                'Indra Jatra': [(8, 25), (9, 10)],     # August-September
                'Maha Shivaratri': [(2, 20), (3, 10)]  # February-March
            },
            
            'festival_multiplier': 1.5
        }
    
    def is_festival_period(self, month, day):
        """Check if current date falls during a major festival"""
        for festival, date_ranges in self.tourism_patterns['festival_dates'].items():
            for start_month, start_day in date_ranges:
                # Simple festival period check (7 days around festival)
                if month == start_month and abs(day - start_day) <= 7:
                    return True
        return False
    
    def get_season_multiplier(self, month):
        """Get seasonal multiplier based on Nepal tourism patterns"""
        if month in [10, 11, 4, 5]:  # Peak season
            return self.tourism_patterns['peak_season_multiplier']
        elif month in [3, 9, 12]:    # Shoulder season
            return self.tourism_patterns['shoulder_season_multiplier']
        else:  # Low season (monsoon)
            return self.tourism_patterns['low_season_multiplier']
    
    def calculate_tourism_based_crowd(self, place, time_slot, day_of_week, month, 
                                    day, is_weekend, weather_condition):
        """Calculate crowd level based on Nepal tourism patterns"""
        
        # Base crowd level for the category
        base_level = self.tourism_patterns['category_base_levels'].get(place.category, 50)
        
        # Apply district multiplier
        district_mult = self.tourism_patterns['district_multipliers'].get(place.district, 1.0)
        
        # Apply monthly tourism pattern
        monthly_mult = self.tourism_patterns['monthly_patterns'].get(month, 0.7)
        
        # Apply seasonal multiplier
        seasonal_mult = self.get_season_multiplier(month)
        
        # Apply weekend/weekday pattern
        day_mult = self.tourism_patterns['weekend_multiplier'] if is_weekend else self.tourism_patterns['weekday_multiplier']
        
        # Apply time slot pattern
        time_mult = self.tourism_patterns['time_slot_patterns'].get(time_slot, 1.0)
        
        # Apply weather multiplier
        weather_mult = self.tourism_patterns['weather_multipliers'].get(weather_condition, 1.0)
        
        # Apply festival multiplier if applicable
        festival_mult = self.tourism_patterns['festival_multiplier'] if self.is_festival_period(month, day) else 1.0
        
        # Calculate final crowd level
        crowd_level = base_level * district_mult * monthly_mult * seasonal_mult * day_mult * time_mult * weather_mult * festival_mult
        
        # Add some realistic variation (±10%)
        variation = random.uniform(-0.1, 0.1)
        crowd_level *= (1 + variation)
        
        # Ensure crowd level is within reasonable bounds
        crowd_level = max(5, min(95, crowd_level))
        
        return round(crowd_level, 1)
    
    def get_season(self, month):
        """Get season based on Nepal's climate"""
        if month in [12, 1, 2]:
            return 'Winter'
        elif month in [3, 4, 5]:
            return 'Spring'
        elif month in [6, 7, 8]:
            return 'Summer'
        else:
            return 'Autumn'
    
    def generate_tourism_data(self):
        """Generate tourism-based crowd data"""
        print("Starting Nepal Tourism-Based Crowd Data Generation...")
        
        places = Place.objects.all()
        if not places.exists():
            print("No places found in database!")
            return
        
        # Generate data for 12 months (full year cycle)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        
        with open(self.output_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([
                'place_id', 'category', 'district', 'time_slot', 'hour', 
                'day_of_week', 'month', 'day', 'season', 'is_weekend', 'is_holiday', 
                'weather_condition', 'crowdlevel', 'tourist_season', 'festival_period'
            ])
            
            total_rows = 0
            
            for place in places:
                print(f"Processing place: {place.name} ({place.category})")
                
                # Generate data for each day
                current_date = start_date
                while current_date <= end_date:
                    month = current_date.month
                    day = current_date.day
                    day_of_week = current_date.weekday()
                    is_weekend = 1 if day_of_week >= 5 else 0
                    season = self.get_season(month)
                    
                    # Determine tourist season
                    if month in [10, 11, 4, 5]:
                        tourist_season = 'Peak'
                    elif month in [3, 9, 12]:
                        tourist_season = 'Shoulder'
                    else:
                        tourist_season = 'Low'
                    
                    # Check if it's a festival period
                    festival_period = 'Yes' if self.is_festival_period(month, day) else 'No'
                    
                    # Generate data for each time slot
                    time_slots = [
                        ('morning', 9),
                        ('afternoon', 14),
                        ('evening', 18)
                    ]
                    
                    for time_slot, hour in time_slots:
                        # Weather conditions based on season
                        if season == 'Summer':  # Monsoon
                            weather_options = ['Rainy', 'Cloudy', 'Foggy']
                        elif season == 'Winter':
                            weather_options = ['Sunny', 'Cloudy', 'Foggy']
                        else:
                            weather_options = ['Sunny', 'Cloudy', 'Rainy']
                        
                        weather_condition = random.choice(weather_options)
                        
                        # Calculate crowd level based on tourism patterns
                        crowd_level = self.calculate_tourism_based_crowd(
                            place, time_slot, day_of_week, month, day, 
                            is_weekend, weather_condition
                        )
                        
                        # Write to CSV
                        writer.writerow([
                            place.id,
                            place.category,
                            place.district,
                            time_slot,
                            hour,
                            day_of_week,
                            month,
                            day,
                            season,
                            is_weekend,
                            0,  # is_holiday (simplified)
                            weather_condition,
                            crowd_level,
                            tourist_season,
                            festival_period
                        ])
                        
                        total_rows += 1
                    
                    current_date += timedelta(days=1)
        
        print(f"Generated {total_rows} rows of tourism-based crowd data")
        print(f"Data saved to: {self.output_file}")
        
        # Validate the generated data
        self.validate_tourism_data()
    
    def validate_tourism_data(self):
        """Validate the generated tourism data"""
        print("\n=== VALIDATING TOURISM-BASED DATA ===")
        
        import pandas as pd
        df = pd.read_csv(self.output_file)
        
        print(f"Total rows: {len(df)}")
        print(f"Unique places: {df['place_id'].nunique()}")
        print(f"Date range: {df['month'].min()} - {df['month'].max()}")
        
        print("\nCrowd level statistics:")
        print(df['crowdlevel'].describe())
        
        print("\nCrowd levels by tourist season:")
        season_stats = df.groupby('tourist_season')['crowdlevel'].agg(['mean', 'std', 'min', 'max'])
        print(season_stats)
        
        print("\nCrowd levels by category:")
        category_stats = df.groupby('category')['crowdlevel'].agg(['mean', 'std', 'min', 'max'])
        print(category_stats)
        
        print("\nCrowd levels by time slot:")
        time_stats = df.groupby('time_slot')['crowdlevel'].agg(['mean', 'std', 'min', 'max'])
        print(time_stats)
        
        print("\nFestival period impact:")
        festival_stats = df.groupby('festival_period')['crowdlevel'].agg(['mean', 'std', 'min', 'max'])
        print(festival_stats)
        
        print("\nMonthly tourism patterns:")
        monthly_stats = df.groupby('month')['crowdlevel'].mean()
        print(monthly_stats)
        
        print("\nData validation completed!")

if __name__ == "__main__":
    generator = NepalTourismBasedGenerator()
    generator.generate_tourism_data() 