import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

class ImprovedCrowdPredictionModel:
    def __init__(self):
        self.model = None
        self.preprocessor = None
        self.label_encoders = {}
        self.is_trained = False
        self.feature_importance = None
        self.model_performance = {}
        
    def prepare_data(self, csv_file):
        """Prepare data with improved feature engineering"""
        print("Loading and preparing data...")
        df = pd.read_csv(csv_file)
        
        # Create additional features
        df = self.create_features(df)
        
        # Separate categorical and numerical features
        categorical_features = ['category', 'district', 'time_slot', 'season', 'weather_condition']
        numerical_features = ['place_id', 'hour', 'day_of_week', 'month', 'is_weekend', 'is_holiday']
        
        # Create target variable
        y = df['crowdlevel']
        
        # Create feature matrix
        X = df[categorical_features + numerical_features]
        
        return X, y, categorical_features, numerical_features
    
    def create_features(self, df):
        """Create additional features for better prediction"""
        # Create time-based features
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        # Create interaction features
        df['weekend_holiday'] = df['is_weekend'] * df['is_holiday']
        df['tourist_season'] = ((df['month'] >= 6) & (df['month'] <= 8)).astype(int)
        df['off_season'] = ((df['month'] >= 12) | (df['month'] <= 2)).astype(int)
        
        # Create time slot features
        df['is_morning'] = (df['time_slot'] == 'morning').astype(int)
        df['is_afternoon'] = (df['time_slot'] == 'afternoon').astype(int)
        df['is_evening'] = (df['time_slot'] == 'evening').astype(int)
        
        # Create weather severity
        weather_severity = {
            'Sunny': 1,
            'Cloudy': 2,
            'Foggy': 3,
            'Rainy': 4
        }
        df['weather_severity'] = df['weather_condition'].map(weather_severity)
        
        return df
    
    def create_preprocessor(self, categorical_features, numerical_features):
        """Create preprocessing pipeline"""
        # Categorical preprocessing
        categorical_transformer = Pipeline([
            ('onehot', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
        ])
        
        # Numerical preprocessing
        numerical_transformer = Pipeline([
            ('scaler', StandardScaler())
        ])
        
        # Combine transformers
        preprocessor = ColumnTransformer(
            transformers=[
                ('cat', categorical_transformer, categorical_features),
                ('num', numerical_transformer, numerical_features)
            ],
            remainder='passthrough'
        )
        
        return preprocessor
    
    def select_best_model(self, X, y):
        """Select the best model using cross-validation"""
        print("Selecting best model...")
        
        # Define models to test
        models = {
            'RandomForest': RandomForestRegressor(random_state=42),
            'GradientBoosting': GradientBoostingRegressor(random_state=42),
            'LinearRegression': LinearRegression()
        }
        
        # Define preprocessing
        categorical_features = ['category', 'district', 'time_slot', 'season', 'weather_condition']
        numerical_features = ['place_id', 'hour', 'day_of_week', 'month', 'is_weekend', 'is_holiday']
        
        preprocessor = self.create_preprocessor(categorical_features, numerical_features)
        
        best_score = -np.inf
        best_model_name = None
        best_model = None
        
        for name, model in models.items():
            # Create pipeline
            pipeline = Pipeline([
                ('preprocessor', preprocessor),
                ('regressor', model)
            ])
            
            # Cross-validation
            scores = cross_val_score(pipeline, X, y, cv=5, scoring='r2')
            mean_score = scores.mean()
            
            print(f"{name}: R² = {mean_score:.4f} (+/- {scores.std() * 2:.4f})")
            
            if mean_score > best_score:
                best_score = mean_score
                best_model_name = name
                best_model = pipeline
        
        print(f"\nBest model: {best_model_name} with R² = {best_score:.4f}")
        return best_model, best_model_name
    
    def train(self, csv_file):
        """Train the improved model"""
        print("Starting improved model training...")
        
        # Prepare data
        X, y, categorical_features, numerical_features = self.prepare_data(csv_file)
        
        # Select best model
        self.model, model_name = self.select_best_model(X, y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=pd.cut(y, bins=10, labels=False)
        )
        
        print(f"Training {model_name}...")
        self.model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = self.model.predict(X_test)
        
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        self.model_performance = {
            'mse': mse,
            'mae': mae,
            'r2': r2,
            'model_name': model_name
        }
        
        print(f"\nModel Performance:")
        print(f"Mean Squared Error: {mse:.2f}")
        print(f"Mean Absolute Error: {mae:.2f}")
        print(f"R² Score: {r2:.4f}")
        
        # Get feature importance if available
        if hasattr(self.model.named_steps['regressor'], 'feature_importances_'):
            feature_names = self.model.named_steps['preprocessor'].get_feature_names_out()
            self.feature_importance = pd.DataFrame({
                'feature': feature_names,
                'importance': self.model.named_steps['regressor'].feature_importances_
            }).sort_values('importance', ascending=False)
            
            print(f"\nTop 10 Most Important Features:")
            print(self.feature_importance.head(10))
        
        self.is_trained = True
        
        # Save model
        self.save_model()
        
        return mse, mae, r2
    
    def predict(self, place_id, category, district, time_slot, day_of_week, 
                month, season, is_weekend, is_holiday, weather_condition, hour=None):
        """Predict crowd level with improved features"""
        if not self.is_trained:
            raise Exception("Model not trained. Please train the model first.")
        
        # Set default hour if not provided
        if hour is None:
            if time_slot == 'morning':
                hour = 9
            elif time_slot == 'afternoon':
                hour = 14
            else:  # evening
                hour = 19
        
        # Create feature dictionary
        features = {
            'place_id': place_id,
            'category': category,
            'district': district,
            'time_slot': time_slot,
            'hour': hour,
            'day_of_week': day_of_week,
            'month': month,
            'season': season,
            'is_weekend': is_weekend,
            'is_holiday': is_holiday,
            'weather_condition': weather_condition
        }
        
        # Create DataFrame with single row
        df = pd.DataFrame([features])
        
        # Create additional features
        df = self.create_features(df)
        
        # Make prediction
        prediction = self.model.predict(df)[0]
        
        # Ensure prediction is within bounds
        prediction = max(0, min(100, prediction))
        
        return round(prediction, 1)
    
    def save_model(self):
        """Save the trained model"""
        model_data = {
            'model': self.model,
            'is_trained': self.is_trained,
            'model_performance': self.model_performance,
            'feature_importance': self.feature_importance
        }
        joblib.dump(model_data, 'improved_crowd_prediction_model.pkl')
        print("Improved model saved successfully!")
    
    def load_model(self):
        """Load a trained model"""
        if os.path.exists('improved_crowd_prediction_model.pkl'):
            model_data = joblib.load('improved_crowd_prediction_model.pkl')
            self.model = model_data['model']
            self.is_trained = model_data['is_trained']
            self.model_performance = model_data.get('model_performance', {})
            self.feature_importance = model_data.get('feature_importance', None)
            print("Improved model loaded successfully!")
            return True
        else:
            print("No saved improved model found!")
            return False
    
    def get_model_info(self):
        """Get information about the trained model"""
        if not self.is_trained:
            return "Model not trained"
        
        info = {
            'model_name': self.model_performance.get('model_name', 'Unknown'),
            'r2_score': self.model_performance.get('r2', 0),
            'mae': self.model_performance.get('mae', 0),
            'mse': self.model_performance.get('mse', 0)
        }
        
        return info

# Utility functions
def get_current_weather():
    """Get current weather condition (placeholder)"""
    import random
    conditions = ['Sunny', 'Cloudy', 'Rainy', 'Foggy']
    return random.choice(conditions)

def get_current_season():
    """Get current season"""
    from datetime import datetime
    month = datetime.now().month
    
    if month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    elif month in [9, 10, 11]:
        return 'Autumn'
    else:
        return 'Winter' 