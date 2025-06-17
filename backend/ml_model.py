import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import os

class CrowdPredictionModel:
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.is_trained = False
        
    def prepare_data(self, csv_file):
        """Prepare data for training"""
        df = pd.read_csv(csv_file)
        
        # Encode categorical variables
        categorical_columns = ['category', 'district', 'time_slot', 'season', 'weather_condition']
        
        for col in categorical_columns:
            if col in df.columns:
                le = LabelEncoder()
                df[f'{col}_encoded'] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
        
        # Create feature matrix
        feature_columns = [
            'place_id', 'category_encoded', 'district_encoded', 'time_slot_encoded',
            'day_of_week', 'month', 'season_encoded', 'is_weekend', 'is_holiday',
            'weather_condition_encoded'
        ]
        
        X = df[feature_columns]
        y = df['crowdlevel']
        
        return X, y
    
    def train(self, csv_file):
        """Train the model"""
        print("Preparing data...")
        X, y = self.prepare_data(csv_file)
        
        print("Splitting data...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        print("Training model...")
        self.model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = self.model.predict(X_test)
        
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"Model Performance:")
        print(f"Mean Squared Error: {mse:.2f}")
        print(f"Mean Absolute Error: {mae:.2f}")
        print(f"R² Score: {r2:.2f}")
        
        self.is_trained = True
        
        # Save model
        self.save_model()
        
        return mse, mae, r2
    
    def predict(self, place_id, category, district, time_slot, day_of_week, 
                month, season, is_weekend, is_holiday, weather_condition):
        """Predict crowd level for given features"""
        if not self.is_trained:
            raise Exception("Model not trained. Please train the model first.")
        
        # Encode categorical features
        features = {
            'place_id': place_id,
            'category_encoded': self.label_encoders['category'].transform([category])[0],
            'district_encoded': self.label_encoders['district'].transform([district])[0],
            'time_slot_encoded': self.label_encoders['time_slot'].transform([time_slot])[0],
            'day_of_week': day_of_week,
            'month': month,
            'season_encoded': self.label_encoders['season'].transform([season])[0],
            'is_weekend': is_weekend,
            'is_holiday': is_holiday,
            'weather_condition_encoded': self.label_encoders['weather_condition'].transform([weather_condition])[0]
        }
        
        # Create feature array
        feature_array = np.array([[
            features['place_id'], features['category_encoded'], features['district_encoded'],
            features['time_slot_encoded'], features['day_of_week'], features['month'],
            features['season_encoded'], features['is_weekend'], features['is_holiday'],
            features['weather_condition_encoded']
        ]])
        
        # Make prediction
        prediction = self.model.predict(feature_array)[0]
        
        # Ensure prediction is within bounds
        prediction = max(0, min(100, prediction))
        
        return round(prediction, 1)
    
    def save_model(self):
        """Save the trained model"""
        model_data = {
            'model': self.model,
            'label_encoders': self.label_encoders,
            'is_trained': self.is_trained
        }
        joblib.dump(model_data, 'crowd_prediction_model.pkl')
        print("Model saved successfully!")
    
    def load_model(self):
        """Load a trained model"""
        if os.path.exists('crowd_prediction_model.pkl'):
            model_data = joblib.load('crowd_prediction_model.pkl')
            self.model = model_data['model']
            self.label_encoders = model_data['label_encoders']
            self.is_trained = model_data['is_trained']
            print("Model loaded successfully!")
            return True
        else:
            print("No saved model found!")
            return False

# Utility function to get current weather (placeholder)
def get_current_weather():
    """Get current weather condition (placeholder - integrate with weather API)"""
    import random
    conditions = ['Sunny', 'Cloudy', 'Rainy', 'Clear']
    return random.choice(conditions)

# Utility function to get season
def get_current_season():
    """Get current season"""
    from datetime import datetime
    month = datetime.now().month
    
    if month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    elif month in [9, 10, 11]:
        return 'Fall'
    else:
        return 'Winter' 