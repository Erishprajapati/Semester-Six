import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.compose import ColumnTransformer
import joblib
import logging
from datetime import datetime, timedelta
import calendar

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RealisticCrowdPredictionModel:
    def __init__(self, model_path='realistic_crowd_model.joblib'):
        self.model_path = model_path
        self.model = None
        self.preprocessor = None
        self.feature_names = None
        self.model_info = {}
        
        # Nepali festival calendar (approximate dates)
        self.nepali_festivals = {
            'Dashain': {'months': [9, 10], 'crowd_multiplier': 3.0},
            'Tihar': {'months': [10, 11], 'crowd_multiplier': 2.5},
            'Buddha_Jayanti': {'months': [4, 5], 'crowd_multiplier': 2.0},
            'Maha_Shivaratri': {'months': [2, 3], 'crowd_multiplier': 2.5},
            'Ram_Navami': {'months': [3, 4], 'crowd_multiplier': 1.8},
            'Krishna_Janmashtami': {'months': [8, 9], 'crowd_multiplier': 1.8},
            'Gai_Jatra': {'months': [7, 8], 'crowd_multiplier': 1.5},
            'Indra_Jatra': {'months': [8, 9], 'crowd_multiplier': 1.5},
            'Bisket_Jatra': {'months': [3, 4], 'crowd_multiplier': 1.8}
        }
        
        # Realistic base crowd levels for different categories
        self.base_crowd_levels = {
            'Religious': {
                'morning': {'weekday': 75, 'weekend': 85, 'festival': 95},
                'afternoon': {'weekday': 45, 'weekend': 60, 'festival': 80},
                'evening': {'weekday': 35, 'weekend': 50, 'festival': 70}
            },
            'Historical': {
                'morning': {'weekday': 30, 'weekend': 50, 'festival': 70},
                'afternoon': {'weekday': 40, 'weekend': 65, 'festival': 80},
                'evening': {'weekday': 25, 'weekend': 40, 'festival': 60}
            },
            'Market': {
                'morning': {'weekday': 60, 'weekend': 75, 'festival': 90},
                'afternoon': {'weekday': 70, 'weekend': 80, 'festival': 95},
                'evening': {'weekday': 50, 'weekend': 65, 'festival': 85}
            },
            'Nature': {
                'morning': {'weekday': 20, 'weekend': 40, 'festival': 60},
                'afternoon': {'weekday': 30, 'weekend': 55, 'festival': 75},
                'evening': {'weekday': 15, 'weekend': 30, 'festival': 50}
            },
            'Cultural': {
                'morning': {'weekday': 25, 'weekend': 45, 'festival': 70},
                'afternoon': {'weekday': 35, 'weekend': 60, 'festival': 80},
                'evening': {'weekday': 20, 'weekend': 35, 'festival': 55}
            }
        }
        
        # Weather impact factors
        self.weather_impact = {
            'Sunny': 1.2,      # Increases crowds
            'Cloudy': 1.0,     # No change
            'Foggy': 0.8,      # Slightly reduces crowds
            'Rainy': 0.5       # Significantly reduces crowds
        }
        
        # Tourist season impact
        self.tourist_season_impact = {
            'Peak': 1.3,       # October-November, April-May
            'Shoulder': 1.1,   # March, September, December
            'Low': 0.9         # June-August
        }
    
    def is_festival_period(self, month, day):
        """Check if current date is during a major festival period"""
        for festival, info in self.nepali_festivals.items():
            if month in info['months']:
                return True, info['crowd_multiplier']
        return False, 1.0
    
    def get_realistic_crowd_level(self, category, time_slot, is_weekend, is_festival, weather, tourist_season, district):
        """Generate realistic crowd levels based on Nepali patterns"""
        
        # Get base crowd level
        if category in self.base_crowd_levels:
            if is_festival:
                base_level = self.base_crowd_levels[category][time_slot]['festival']
            elif is_weekend:
                base_level = self.base_crowd_levels[category][time_slot]['weekend']
            else:
                base_level = self.base_crowd_levels[category][time_slot]['weekday']
        else:
            base_level = 30  # Default for unknown categories
        
        # Apply weather impact
        weather_factor = self.weather_impact.get(weather, 1.0)
        
        # Apply tourist season impact
        tourist_factor = self.tourist_season_impact.get(tourist_season, 1.0)
        
        # District-specific adjustments
        district_factor = 1.0
        if district == 'Kathmandu':
            district_factor = 1.2  # Kathmandu is more crowded
        elif district == 'Bhaktapur':
            district_factor = 1.1  # Bhaktapur is popular with tourists
        elif district == 'Lalitpur':
            district_factor = 1.0  # Lalitpur is moderate
        
        # Calculate final crowd level
        final_level = base_level * weather_factor * tourist_factor * district_factor
        
        # Add some realistic variation (±10%)
        variation = np.random.normal(0, 0.1)
        final_level = final_level * (1 + variation)
        
        # Ensure within valid range
        return max(5, min(100, final_level))
    
    def generate_realistic_training_data(self, output_file='realistic_crowd_data.csv'):
        """Generate realistic training data based on Nepali crowd patterns"""
        logger.info("Generating realistic training data...")
        
        data = []
        
        # Places data (you can expand this)
        places = [
            {'id': 1, 'name': 'Pashupatinath Temple', 'category': 'Religious', 'district': 'Kathmandu'},
            {'id': 2, 'name': 'Swayambhunath Stupa', 'category': 'Religious', 'district': 'Kathmandu'},
            {'id': 3, 'name': 'Boudhanath Stupa', 'category': 'Religious', 'district': 'Kathmandu'},
            {'id': 4, 'name': 'Kathmandu Durbar Square', 'category': 'Historical', 'district': 'Kathmandu'},
            {'id': 5, 'name': 'Asan Market', 'category': 'Market', 'district': 'Kathmandu'},
            {'id': 6, 'name': 'Garden of Dreams', 'category': 'Nature', 'district': 'Kathmandu'},
            {'id': 7, 'name': 'Patan Durbar Square', 'category': 'Historical', 'district': 'Lalitpur'},
            {'id': 8, 'name': 'Bhaktapur Durbar Square', 'category': 'Historical', 'district': 'Bhaktapur'},
            {'id': 9, 'name': 'Nyatapola Temple', 'category': 'Religious', 'district': 'Bhaktapur'},
            {'id': 10, 'name': 'Changu Narayan Temple', 'category': 'Religious', 'district': 'Bhaktapur'},
            {'id': 11, 'name': 'Kopan Monastery', 'category': 'Religious', 'district': 'Kathmandu'},
            {'id': 12, 'name': 'Thamel Market', 'category': 'Market', 'district': 'Kathmandu'},
            {'id': 13, 'name': 'Shivapuri National Park', 'category': 'Nature', 'district': 'Kathmandu'},
            {'id': 14, 'name': 'Golden Temple', 'category': 'Religious', 'district': 'Lalitpur'},
            {'id': 15, 'name': 'Kirtipur Durbar Square', 'category': 'Historical', 'district': 'Kathmandu'}
        ]
        
        # Generate data for each place for a full year
        for place in places:
            for month in range(1, 13):
                for day in range(1, 29):  # Simplified for 28 days per month
                    for time_slot in ['morning', 'afternoon', 'evening']:
                        for weather in ['Sunny', 'Cloudy', 'Foggy', 'Rainy']:
                            # Get day of week
                            try:
                                date_obj = datetime(2024, month, day)
                                day_of_week = date_obj.weekday()
                                is_weekend = 1 if day_of_week >= 5 else 0
                            except:
                                continue
                            
                            # Determine season
                            if month in [3, 4, 5]:
                                season = 'Spring'
                            elif month in [6, 7, 8]:
                                season = 'Summer'
                            elif month in [9, 10, 11]:
                                season = 'Autumn'
                            else:
                                season = 'Winter'
                            
                            # Determine tourist season
                            if month in [10, 11, 4, 5]:
                                tourist_season = 'Peak'
                            elif month in [3, 9, 12]:
                                tourist_season = 'Shoulder'
                            else:
                                tourist_season = 'Low'
                            
                            # Check for festivals
                            is_festival, festival_multiplier = self.is_festival_period(month, day)
                            
                            # Get hour based on time slot
                            hour = 9 if time_slot == 'morning' else 14 if time_slot == 'afternoon' else 18
                            
                            # Generate realistic crowd level
                            crowd_level = self.get_realistic_crowd_level(
                                place['category'], time_slot, is_weekend, is_festival, 
                                weather, tourist_season, place['district']
                            )
                            
                            # Apply festival multiplier if applicable
                            if is_festival:
                                crowd_level = min(100, crowd_level * festival_multiplier)
                            
                            # Add to dataset
                            data.append({
                                'place_id': place['id'],
                                'place_name': place['name'],
                                'category': place['category'],
                                'district': place['district'],
                                'time_slot': time_slot,
                                'hour': hour,
                                'day_of_week': day_of_week,
                                'month': month,
                                'day': day,
                                'season': season,
                                'is_weekend': is_weekend,
                                'is_holiday': 1 if is_festival else 0,
                                'weather_condition': weather,
                                'crowdlevel': round(crowd_level, 1),
                                'tourist_season': tourist_season,
                                'festival_period': 'Yes' if is_festival else 'No'
                            })
        
        # Create DataFrame and save
        df = pd.DataFrame(data)
        df.to_csv(output_file, index=False)
        logger.info(f"Generated {len(df)} realistic data points saved to {output_file}")
        
        return df
    
    def engineer_realistic_features(self, df):
        """Engineer features specific to Nepali crowd patterns"""
        logger.info("Engineering realistic features...")
        
        # Cyclical encoding for time
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        # Prayer time features (important for religious places)
        df['is_prayer_time'] = ((df['hour'] >= 4) & (df['hour'] <= 8)) | ((df['hour'] >= 17) & (df['hour'] <= 19))
        df['is_peak_visiting_hour'] = ((df['hour'] >= 9) & (df['hour'] <= 11)) | ((df['hour'] >= 14) & (df['hour'] <= 16))
        
        # Category-specific features
        df['is_religious'] = (df['category'] == 'Religious').astype(int)
        df['is_historical'] = (df['category'] == 'Historical').astype(int)
        df['is_market'] = (df['category'] == 'Market').astype(int)
        df['is_nature'] = (df['category'] == 'Nature').astype(int)
        df['is_cultural'] = (df['category'] == 'Cultural').astype(int)
        
        # District features
        df['is_kathmandu'] = (df['district'] == 'Kathmandu').astype(int)
        df['is_lalitpur'] = (df['district'] == 'Lalitpur').astype(int)
        df['is_bhaktapur'] = (df['district'] == 'Bhaktapur').astype(int)
        
        # Weather severity
        weather_severity = {'Sunny': 1, 'Cloudy': 2, 'Foggy': 3, 'Rainy': 4}
        df['weather_severity'] = df['weather_condition'].map(weather_severity)
        
        # Season features
        df['is_spring'] = (df['season'] == 'Spring').astype(int)
        df['is_summer'] = (df['season'] == 'Summer').astype(int)
        df['is_autumn'] = (df['season'] == 'Autumn').astype(int)
        df['is_winter'] = (df['season'] == 'Winter').astype(int)
        
        # Time slot features
        df['is_morning'] = (df['time_slot'] == 'morning').astype(int)
        df['is_afternoon'] = (df['time_slot'] == 'afternoon').astype(int)
        df['is_evening'] = (df['time_slot'] == 'evening').astype(int)
        
        # Tourist season features
        df['is_peak_season'] = (df['tourist_season'] == 'Peak').astype(int)
        df['is_shoulder_season'] = (df['tourist_season'] == 'Shoulder').astype(int)
        df['is_low_season'] = (df['tourist_season'] == 'Low').astype(int)
        
        # Festival features
        df['is_festival'] = (df['festival_period'] == 'Yes').astype(int)
        
        # Interaction features
        df['religious_prayer_time'] = df['is_religious'] * df['is_prayer_time']
        df['religious_weekend'] = df['is_religious'] * df['is_weekend']
        df['market_peak_hour'] = df['is_market'] * df['is_peak_visiting_hour']
        df['festival_weekend'] = df['is_festival'] * df['is_weekend']
        df['peak_season_weekend'] = df['is_peak_season'] * df['is_weekend']
        
        return df
    
    def train_model(self, csv_file='realistic_crowd_data.csv'):
        """Train the realistic crowd prediction model"""
        logger.info("Training realistic crowd prediction model...")
        
        try:
            # Load or generate data
            if not os.path.exists(csv_file):
                logger.info("Training data not found. Generating realistic data...")
                df = self.generate_realistic_training_data(csv_file)
            else:
                df = pd.read_csv(csv_file)
            
            # Engineer features
            df = self.engineer_realistic_features(df)
            
            # Prepare features and target
            feature_columns = [col for col in df.columns if col not in ['crowdlevel', 'place_name']]
            X = df[feature_columns]
            y = df['crowdlevel']
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Train model
            self.model = RandomForestRegressor(n_estimators=200, random_state=42, max_depth=15)
            self.model.fit(X_train, y_train)
            
            # Evaluate
            y_pred = self.model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            logger.info(f"Model Performance:")
            logger.info(f"R² Score: {r2:.4f}")
            logger.info(f"Mean Absolute Error: {mae:.2f}")
            logger.info(f"Mean Squared Error: {mse:.2f}")
            
            # Save model
            self.save_model()
            
            # Test realistic predictions
            self.test_realistic_predictions()
            
            return True
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            return False
    
    def test_realistic_predictions(self):
        """Test the model with realistic scenarios"""
        logger.info("Testing realistic predictions...")
        
        test_cases = [
            {
                'name': 'Pashupatinath Temple - Sunday Morning Prayer',
                'params': {
                    'place_id': 1, 'category': 'Religious', 'district': 'Kathmandu',
                    'time_slot': 'morning', 'hour': 6, 'day_of_week': 6, 'month': 10,
                    'day': 15, 'season': 'Autumn', 'is_weekend': 1, 'is_holiday': 0,
                    'weather_condition': 'Sunny', 'tourist_season': 'Peak', 'festival_period': 'No'
                }
            },
            {
                'name': 'Asan Market - Saturday Afternoon',
                'params': {
                    'place_id': 5, 'category': 'Market', 'district': 'Kathmandu',
                    'time_slot': 'afternoon', 'hour': 14, 'day_of_week': 5, 'month': 11,
                    'day': 20, 'season': 'Autumn', 'is_weekend': 1, 'is_holiday': 0,
                    'weather_condition': 'Cloudy', 'tourist_season': 'Peak', 'festival_period': 'No'
                }
            },
            {
                'name': 'Garden of Dreams - Rainy Weekday',
                'params': {
                    'place_id': 6, 'category': 'Nature', 'district': 'Kathmandu',
                    'time_slot': 'afternoon', 'hour': 14, 'day_of_week': 2, 'month': 7,
                    'day': 10, 'season': 'Summer', 'is_weekend': 0, 'is_holiday': 0,
                    'weather_condition': 'Rainy', 'tourist_season': 'Low', 'festival_period': 'No'
                }
            }
        ]
        
        for test_case in test_cases:
            try:
                prediction = self.predict(**test_case['params'])
                logger.info(f"🕍 {test_case['name']}: {prediction:.1f}% crowd level")
            except Exception as e:
                logger.error(f"❌ {test_case['name']}: Error - {e}")
    
    def predict(self, **kwargs):
        """Make realistic crowd predictions"""
        if self.model is None:
            raise ValueError("Model not loaded. Please train the model first.")
        
        # Create feature vector
        features = self.engineer_realistic_features(pd.DataFrame([kwargs]))
        
        # Remove target column if present
        if 'crowdlevel' in features.columns:
            features = features.drop('crowdlevel', axis=1)
        if 'place_name' in features.columns:
            features = features.drop('place_name', axis=1)
        
        # Make prediction
        prediction = self.model.predict(features)[0]
        
        # Ensure valid range
        return max(5, min(100, prediction))
    
    def save_model(self):
        """Save the trained model"""
        model_data = {
            'model': self.model,
            'feature_names': self.feature_names,
            'model_info': self.model_info
        }
        joblib.dump(model_data, self.model_path)
        logger.info(f"Model saved to {self.model_path}")
    
    def load_model(self):
        """Load the trained model"""
        try:
            model_data = joblib.load(self.model_path)
            self.model = model_data['model']
            self.feature_names = model_data.get('feature_names')
            self.model_info = model_data.get('model_info', {})
            logger.info("Model loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False

def main():
    """Main function to train the realistic model"""
    model = RealisticCrowdPredictionModel()
    
    # Generate realistic training data and train model
    success = model.train_model()
    
    if success:
        logger.info("✅ Realistic crowd prediction model trained successfully!")
        logger.info("🎯 The model now reflects real Nepali crowd patterns:")
        logger.info("   • Temples are crowded during morning prayers")
        logger.info("   • Markets are busy on weekends")
        logger.info("   • Festivals significantly increase crowds")
        logger.info("   • Weather affects crowd levels realistically")
    else:
        logger.error("❌ Model training failed!")

if __name__ == "__main__":
    main() 