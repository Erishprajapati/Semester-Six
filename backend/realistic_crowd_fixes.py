import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RealisticCrowdFixer:
    def __init__(self):
        # Realistic base crowd levels for Nepali context
        self.realistic_patterns = {
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
            }
        }
        
        # Nepali festivals (approximate months)
        self.festivals = {
            'Dashain': [9, 10],      # High crowds everywhere
            'Tihar': [10, 11],       # High crowds, especially temples
            'Buddha_Jayanti': [4, 5], # High crowds at Buddhist sites
            'Maha_Shivaratri': [2, 3], # High crowds at Shiva temples
            'Ram_Navami': [3, 4],    # Moderate crowds
            'Krishna_Janmashtami': [8, 9], # Moderate crowds
        }
    
    def fix_crowd_data(self, input_file='nepal_tourism_crowd_data.csv', output_file='fixed_crowd_data.csv'):
        """Fix the crowd data to reflect realistic Nepali patterns"""
        logger.info("Loading and fixing crowd data...")
        
        # Load original data
        df = pd.read_csv(input_file)
        logger.info(f"Loaded {len(df)} records")
        
        # Apply realistic fixes
        df = self.apply_realistic_fixes(df)
        
        # Save fixed data
        df.to_csv(output_file, index=False)
        logger.info(f"Fixed data saved to {output_file}")
        
        return df
    
    def apply_realistic_fixes(self, df):
        """Apply realistic crowd level fixes"""
        logger.info("Applying realistic crowd fixes...")
        
        for idx, row in df.iterrows():
            category = row['category']
            time_slot = row['time_slot']
            is_weekend = row['is_weekend']
            month = row['month']
            weather = row['weather_condition']
            
            # Get base realistic level
            if category in self.realistic_patterns:
                if is_weekend:
                    base_level = self.realistic_patterns[category][time_slot]['weekend']
                else:
                    base_level = self.realistic_patterns[category][time_slot]['weekday']
            else:
                base_level = 40  # Default
            
            # Apply festival multiplier
            festival_multiplier = self.get_festival_multiplier(month, category)
            
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
    
    def get_festival_multiplier(self, month, category):
        """Get festival multiplier based on month and category"""
        for festival, months in self.festivals.items():
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

def main():
    """Main function to fix crowd data"""
    fixer = RealisticCrowdFixer()
    
    # Fix the crowd data
    fixed_df = fixer.fix_crowd_data()
    
    # Show some examples
    logger.info("\n=== Fixed Crowd Level Examples ===")
    
    # Religious places
    religious_data = fixed_df[fixed_df['category'] == 'Religious'].head(5)
    for _, row in religious_data.iterrows():
        logger.info(f"🕍 {row['time_slot']} {'(Weekend)' if row['is_weekend'] else '(Weekday)'}: {row['crowdlevel']}%")
    
    # Market places
    market_data = fixed_df[fixed_df['category'] == 'Market'].head(5)
    for _, row in market_data.iterrows():
        logger.info(f"🛒 {row['time_slot']} {'(Weekend)' if row['is_weekend'] else '(Weekday)'}: {row['crowdlevel']}%")
    
    logger.info("\n✅ Crowd data fixed with realistic Nepali patterns!")
    logger.info("🎯 Key improvements:")
    logger.info("   • Temples now show high crowds during morning prayers")
    logger.info("   • Markets are busy on weekends")
    logger.info("   • Festivals significantly increase crowds")
    logger.info("   • Weather realistically affects crowd levels")

if __name__ == "__main__":
    main() 