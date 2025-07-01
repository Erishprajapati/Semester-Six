from django.core.management.base import BaseCommand
from django.conf import settings
import pandas as pd
import numpy as np
import os
import logging

logger = logging.getLogger(__name__)

class Command(BaseCommand):
    help = 'Fix crowd data to reflect realistic Nepali crowd patterns'

    def add_arguments(self, parser):
        parser.add_argument(
            '--input-file',
            type=str,
            default='nepal_tourism_crowd_data.csv',
            help='Input CSV file path'
        )
        parser.add_argument(
            '--output-file',
            type=str,
            default='realistic_crowd_data.csv',
            help='Output CSV file path'
        )
        parser.add_argument(
            '--backup',
            action='store_true',
            help='Create backup of original file'
        )

    def handle(self, *args, **options):
        self.stdout.write(self.style.SUCCESS('🚀 Starting crowd data fix...'))
        
        input_file = options['input_file']
        output_file = options['output_file']
        
        # Check if input file exists
        if not os.path.exists(input_file):
            self.stdout.write(self.style.ERROR(f'❌ Input file {input_file} not found!'))
            return
        
        # Create backup if requested
        if options['backup']:
            backup_file = f"{input_file}.backup"
            import shutil
            shutil.copy2(input_file, backup_file)
            self.stdout.write(self.style.SUCCESS(f'📦 Backup created: {backup_file}'))
        
        # Load and fix data
        try:
            df = self.fix_crowd_data(input_file)
            
            # Save fixed data
            df.to_csv(output_file, index=False)
            
            self.stdout.write(self.style.SUCCESS(f'✅ Fixed data saved to {output_file}'))
            self.stdout.write(self.style.SUCCESS(f'📊 Total records processed: {len(df)}'))
            
            # Show examples
            self.show_examples(df)
            
        except Exception as e:
            self.stdout.write(self.style.ERROR(f'❌ Error: {e}'))
    
    def fix_crowd_data(self, input_file):
        """Fix crowd data with realistic Nepali patterns"""
        self.stdout.write('📖 Loading data...')
        df = pd.read_csv(input_file)
        
        # Realistic patterns for Nepali context
        realistic_patterns = {
            'Religious': {
                'morning': {'weekday': 80, 'weekend': 90, 'festival': 95},
                'afternoon': {'weekday': 50, 'weekend': 70, 'festival': 85},
                'evening': {'weekday': 40, 'weekend': 60, 'festival': 75}
            },
            'Historical': {
                'morning': {'weekday': 35, 'weekend': 60, 'festival': 80},
                'afternoon': {'weekday': 45, 'weekend': 75, 'festival': 85},
                'evening': {'weekday': 30, 'weekend': 50, 'festival': 70}
            },
            'Market': {
                'morning': {'weekday': 70, 'weekend': 85, 'festival': 95},
                'afternoon': {'weekday': 80, 'weekend': 90, 'festival': 98},
                'evening': {'weekday': 60, 'weekend': 75, 'festival': 90}
            },
            'Nature': {
                'morning': {'weekday': 25, 'weekend': 50, 'festival': 70},
                'afternoon': {'weekday': 35, 'weekend': 65, 'festival': 80},
                'evening': {'weekday': 20, 'weekend': 40, 'festival': 60}
            },
            'Cultural': {
                'morning': {'weekday': 30, 'weekend': 55, 'festival': 75},
                'afternoon': {'weekday': 40, 'weekend': 70, 'festival': 85},
                'evening': {'weekday': 25, 'weekend': 45, 'festival': 65}
            }
        }
        
        # Nepali festivals
        festivals = {
            'Dashain': [9, 10],      # High crowds everywhere
            'Tihar': [10, 11],       # High crowds, especially temples
            'Buddha_Jayanti': [4, 5], # High crowds at Buddhist sites
            'Maha_Shivaratri': [2, 3], # High crowds at Shiva temples
            'Ram_Navami': [3, 4],    # Moderate crowds
            'Krishna_Janmashtami': [8, 9], # Moderate crowds
        }
        
        self.stdout.write('🔧 Applying realistic fixes...')
        
        for idx, row in df.iterrows():
            category = row['category']
            time_slot = row['time_slot']
            is_weekend = row['is_weekend']
            month = row['month']
            weather = row['weather_condition']
            
            # Get base realistic level
            if category in realistic_patterns:
                if is_weekend:
                    base_level = realistic_patterns[category][time_slot]['weekend']
                else:
                    base_level = realistic_patterns[category][time_slot]['weekday']
            else:
                base_level = 40  # Default
            
            # Apply festival multiplier
            festival_multiplier = self.get_festival_multiplier(month, category, festivals)
            
            # Apply weather impact
            weather_multiplier = self.get_weather_multiplier(weather)
            
            # Apply tourist season impact
            tourist_multiplier = self.get_tourist_multiplier(month)
            
            # Calculate new crowd level
            new_level = base_level * festival_multiplier * weather_multiplier * tourist_multiplier
            
            # Add some variation (±10%)
            variation = np.random.normal(0, 0.1)
            new_level = new_level * (1 + variation)
            
            # Ensure valid range
            new_level = max(5, min(100, new_level))
            
            # Update the crowd level
            df.at[idx, 'crowdlevel'] = round(new_level, 1)
        
        return df
    
    def get_festival_multiplier(self, month, category, festivals):
        """Get festival multiplier"""
        for festival, months in festivals.items():
            if month in months:
                if festival in ['Dashain', 'Tihar']:
                    return 2.5  # Major festivals
                elif festival == 'Buddha_Jayanti' and category == 'Religious':
                    return 2.0  # Buddhist sites
                elif festival == 'Maha_Shivaratri' and category == 'Religious':
                    return 2.0  # Shiva temples
                else:
                    return 1.5  # Other festivals
        return 1.0  # No festival
    
    def get_weather_multiplier(self, weather):
        """Get weather impact multiplier"""
        weather_effects = {
            'Sunny': 1.2,      # Good weather increases crowds
            'Cloudy': 1.0,     # No change
            'Foggy': 0.8,      # Slightly reduces crowds
            'Rainy': 0.5       # Significantly reduces crowds
        }
        return weather_effects.get(weather, 1.0)
    
    def get_tourist_multiplier(self, month):
        """Get tourist season multiplier"""
        if month in [10, 11, 4, 5]:  # Peak season
            return 1.3
        elif month in [3, 9, 12]:    # Shoulder season
            return 1.1
        else:                         # Low season
            return 0.9
    
    def show_examples(self, df):
        """Show examples of fixed crowd levels"""
        self.stdout.write('\n📊 Examples of Fixed Crowd Levels:')
        
        # Religious places
        religious_data = df[df['category'] == 'Religious'].head(3)
        self.stdout.write('\n🕍 Religious Places:')
        for _, row in religious_data.iterrows():
            day_type = 'Weekend' if row['is_weekend'] else 'Weekday'
            self.stdout.write(f"   {row['time_slot']} ({day_type}): {row['crowdlevel']}%")
        
        # Market places
        market_data = df[df['category'] == 'Market'].head(3)
        self.stdout.write('\n🛒 Market Places:')
        for _, row in market_data.iterrows():
            day_type = 'Weekend' if row['is_weekend'] else 'Weekday'
            self.stdout.write(f"   {row['time_slot']} ({day_type}): {row['crowdlevel']}%")
        
        # Historical places
        historical_data = df[df['category'] == 'Historical'].head(3)
        self.stdout.write('\n🏛️ Historical Places:')
        for _, row in historical_data.iterrows():
            day_type = 'Weekend' if row['is_weekend'] else 'Weekday'
            self.stdout.write(f"   {row['time_slot']} ({day_type}): {row['crowdlevel']}%")
        
        self.stdout.write('\n🎯 Key Improvements:')
        self.stdout.write('   • Temples now show realistic morning prayer crowds')
        self.stdout.write('   • Markets are busy on weekends')
        self.stdout.write('   • Festivals significantly increase crowds')
        self.stdout.write('   • Weather realistically affects crowd levels') 