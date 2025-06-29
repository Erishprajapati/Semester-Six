import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import os
import warnings
import logging
from datetime import datetime, timedelta
import argparse

# Try to import XGBoost and SHAP (optional dependencies)
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost not available. Install with: pip install xgboost")

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("SHAP not available. Install with: pip install shap")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ImprovedCrowdPredictionModel:
    def __init__(self, model_path='crowd_prediction_model.joblib'):
        self.model_path = model_path
        self.model = None
        self.preprocessor = None
        self.feature_names = None
        self.model_info = {}
        
    def clean_data(self, df):
        """Enhanced data cleaning with better outlier detection"""
        logger.info("Starting enhanced data cleaning...")
        
        initial_rows = len(df)
        
        # Remove missing values
        df = df.dropna()
        logger.info(f"Removed {initial_rows - len(df)} rows with missing values")
        
        # Validate crowd levels
        invalid_crowd = df[(df['crowdlevel'] < 0) | (df['crowdlevel'] > 100)]
        if len(invalid_crowd) > 0:
            df = df[(df['crowdlevel'] >= 0) & (df['crowdlevel'] <= 100)]
            logger.info(f"Removed {len(invalid_crowd)} rows with invalid crowd levels. Remaining rows: {len(df)}")
        
        # Enhanced outlier detection using IQR method
        Q1 = df['crowdlevel'].quantile(0.25)
        Q3 = df['crowdlevel'].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = df[(df['crowdlevel'] < lower_bound) | (df['crowdlevel'] > upper_bound)]
        if len(outliers) > 0:
            df = df[(df['crowdlevel'] >= lower_bound) & (df['crowdlevel'] <= upper_bound)]
            logger.info(f"Removed {len(outliers)} outliers using IQR method")
        
        # Remove duplicates
        initial_rows = len(df)
        df = df.drop_duplicates()
        logger.info(f"Removed {initial_rows - len(df)} duplicate rows")
        
        return df
    
    def engineer_features(self, df):
        """Enhanced feature engineering with more sophisticated features"""
        logger.info("Engineering advanced features...")
        
        # Cyclical encoding for time-based features
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        # Enhanced interaction features
        df['is_busy_day'] = df['is_weekend'] * df['is_holiday']
        df['is_peak_hour'] = ((df['hour'] >= 9) & (df['hour'] <= 11)) | ((df['hour'] >= 17) & (df['hour'] <= 19))
        df['is_off_peak'] = (df['hour'] >= 22) | (df['hour'] <= 6)
        
        # Tourist season features (from nepal_tourism_crowd_data.csv)
        if 'tourist_season' in df.columns:
            df['is_peak_season'] = (df['tourist_season'] == 'Peak').astype(int)
            df['is_shoulder_season'] = (df['tourist_season'] == 'Shoulder').astype(int)
            df['is_low_season'] = (df['tourist_season'] == 'Low').astype(int)
        else:
            # Fallback logic based on month
            df['is_peak_season'] = df['month'].isin([10, 11, 4, 5]).astype(int)
            df['is_shoulder_season'] = df['month'].isin([3, 9, 12]).astype(int)
            df['is_low_season'] = df['month'].isin([6, 7, 8]).astype(int)
        
        # Festival period features
        if 'festival_period' in df.columns:
            df['is_festival'] = (df['festival_period'] == 'Yes').astype(int)
        else:
            df['is_festival'] = 0
        
        # Category-specific features
        df['is_religious'] = (df['category'] == 'Religious').astype(int)
        df['is_historical'] = (df['category'] == 'Historical').astype(int)
        df['is_nature'] = (df['category'].isin(['Nature', 'Natural', 'Park'])).astype(int)
        df['is_market'] = (df['category'] == 'Market').astype(int)
        df['is_cultural'] = (df['category'] == 'Cultural').astype(int)
        
        # District features
        df['is_kathmandu'] = (df['district'] == 'Kathmandu').astype(int)
        df['is_lalitpur'] = (df['district'] == 'Lalitpur').astype(int)
        df['is_bhaktapur'] = (df['district'] == 'Bhaktapur').astype(int)
        
        # Weather severity mapping
        weather_severity = {
            'Sunny': 1,
            'Cloudy': 2,
            'Foggy': 3,
            'Rainy': 4
        }
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
        
        # Advanced interaction features
        df['weekend_morning'] = df['is_weekend'] * df['is_morning']
        df['weekend_evening'] = df['is_weekend'] * df['is_evening']
        df['peak_season_weekend'] = df['is_peak_season'] * df['is_weekend']
        df['religious_weekend'] = df['is_religious'] * df['is_weekend']
        df['market_peak_hour'] = df['is_market'] * df['is_peak_hour']
        
        # Crowd level bins for feature engineering (only if crowdlevel exists)
        if 'crowdlevel' in df.columns:
            df['crowd_level_bin'] = pd.cut(df['crowdlevel'], bins=[0, 30, 70, 100], labels=['Low', 'Medium', 'High'])
        
        logger.info(f"Engineered {len([col for col in df.columns if col not in ['place_id', 'category', 'district', 'time_slot', 'hour', 'day_of_week', 'month', 'day', 'season', 'is_weekend', 'is_holiday', 'weather_condition', 'crowdlevel', 'tourist_season', 'festival_period']])} total features")
        
        return df
    
    def prepare_data(self, df):
        """Prepare data for training with enhanced preprocessing"""
        logger.info("Preparing data for training...")
        
        # Separate features and target
        feature_columns = [col for col in df.columns if col not in ['crowdlevel', 'crowd_level_bin']]
        X = df[feature_columns]
        y = df['crowdlevel']
        
        # Define categorical and numerical features
        categorical_features = ['category', 'district', 'time_slot', 'season', 'weather_condition']
        if 'tourist_season' in df.columns:
            categorical_features.append('tourist_season')
        if 'festival_period' in df.columns:
            categorical_features.append('festival_period')
        
        numerical_features = [col for col in X.columns if col not in categorical_features]
        
        # Create preprocessing pipeline
        numerical_transformer = StandardScaler()
        categorical_transformer = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
        
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_features),
                ('cat', categorical_transformer, categorical_features)
            ],
            remainder='drop'
        )
        
        # Fit and transform the data
        X_processed = self.preprocessor.fit_transform(X)
        
        # Get feature names
        feature_names = []
        feature_names.extend(numerical_features)
        
        # Add categorical feature names
        for i, feature in enumerate(categorical_features):
            if feature in X.columns:
                categories = self.preprocessor.named_transformers_['cat'].categories_[i][1:]
                feature_names.extend([f"{feature}_{cat}" for cat in categories])
        
        self.feature_names = feature_names
        
        logger.info(f"Prepared data: {X_processed.shape[0]} samples, {X_processed.shape[1]} features")
        
        return X_processed, y
    
    def train_model(self, csv_file='nepal_tourism_crowd_data.csv'):
        """Train the enhanced model with cross-validation and model selection"""
        logger.info("🚀 Starting enhanced crowd prediction model training...")
        logger.info("🧠 Training enhanced model with advanced features...")
        
        try:
            # Load data
            logger.info(f"Loading data from {csv_file}...")
            df = pd.read_csv(csv_file)
            
            # Clean data
            df = self.clean_data(df)
            
            # Engineer features
            df = self.engineer_features(df)
            
            # Prepare data
            X, y = self.prepare_data(df)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Define models for comparison
            models = {
                'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
                'GradientBoosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
            }
            
            if XGBOOST_AVAILABLE:
                models['XGBoost'] = xgb.XGBRegressor(n_estimators=100, random_state=42)
            
            # Cross-validation to select best model
            logger.info("Selecting best model using 5-fold cross-validation...")
            best_model_name = None
            best_score = -np.inf
            
            for name, model in models.items():
                cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
                mean_score = cv_scores.mean()
                std_score = cv_scores.std()
                logger.info(f"{name}: R² = {mean_score:.4f} (+/- {std_score:.4f})")
                
                if mean_score > best_score:
                    best_score = mean_score
                    best_model_name = name
            
            logger.info(f"\nBest model: {best_model_name} with R² = {best_score:.4f}")
            
            # Train the best model
            logger.info(f"Training {best_model_name}...")
            self.model = models[best_model_name]
            self.model.fit(X_train, y_train)
            
            # Evaluate on test set
            y_pred = self.model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            logger.info(f"\nModel Performance:")
            logger.info(f"Mean Squared Error: {mse:.2f}")
            logger.info(f"Mean Absolute Error: {mae:.2f}")
            logger.info(f"R² Score: {r2:.4f}")
            
            # Feature importance
            if hasattr(self.model, 'feature_importances_'):
                feature_importance = self.model.feature_importances_
                feature_importance_df = pd.DataFrame({
                    'feature': self.feature_names,
                    'importance': feature_importance
                }).sort_values('importance', ascending=False)
                
                logger.info(f"\nTop 10 Most Important Features:")
                for i, row in feature_importance_df.head(10).iterrows():
                    logger.info(f"  {row['feature']}: {row['importance']:.4f}")
            
            # Save model info
            self.model_info = {
                'model_name': best_model_name,
                'r2_score': r2,
                'mae': mae,
                'mse': mse,
                'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'data_source': csv_file,
                'total_samples': len(df),
                'total_features': X.shape[1],
                'shap_available': SHAP_AVAILABLE
            }
            
            # Save the model
            self.save_model()
            
            # Test predictions
            logger.info("\n🧪 Testing predictions...")
            test_cases = [
                {'name': 'Temple, Sunday morning, sunny', 'params': {'place_id': 1, 'category': 'Religious', 'district': 'Kathmandu', 'time_slot': 'morning', 'day_of_week': 6, 'month': 12, 'season': 'Winter', 'is_weekend': 1, 'is_holiday': 0, 'weather_condition': 'Sunny'}},
                {'name': 'Market, Wednesday afternoon, cloudy', 'params': {'place_id': 2, 'category': 'Market', 'district': 'Kathmandu', 'time_slot': 'afternoon', 'day_of_week': 2, 'month': 6, 'season': 'Summer', 'is_weekend': 0, 'is_holiday': 0, 'weather_condition': 'Cloudy'}},
                {'name': 'Park, Saturday evening, sunny', 'params': {'place_id': 3, 'category': 'Park', 'district': 'Lalitpur', 'time_slot': 'evening', 'day_of_week': 5, 'month': 4, 'season': 'Spring', 'is_weekend': 1, 'is_holiday': 0, 'weather_condition': 'Sunny'}}
            ]
            
            for test_case in test_cases:
                try:
                    prediction = self.predict(**test_case['params'])
                    logger.info(f"🕍 {test_case['name']}: {prediction:.1f}% crowd level")
                except Exception as e:
                    logger.error(f"❌ {test_case['name']}: Error - {e}")
            
            if hasattr(self.model, 'feature_importances_'):
                logger.info(f"\n🏆 Top 5 Most Important Features:")
                for i, row in feature_importance_df.head(5).iterrows():
                    logger.info(f"  📈 {row['feature']}: {row['importance']:.4f}")
            
            if not SHAP_AVAILABLE:
                logger.info("\n⚠️ SHAP not available. Install with: pip install shap")
            
            logger.info(f"\n📖 Model Usage:")
            logger.info(f"  • The model is automatically used by the API endpoints")
            logger.info(f"  • Use --force-retrain to update the model with new data")
            logger.info(f"  • Check model performance with: python manage.py train_improved_crowd_model")
            
            return True
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            return False
    
    def predict(self, place_id, category, district, time_slot, day_of_week, month, season, is_weekend, is_holiday, weather_condition, hour=None):
        """Make predictions with enhanced feature engineering"""
        if self.model is None:
            raise ValueError("Model not loaded. Please train or load the model first.")
        
        # Engineer features for prediction
        logger.info("Engineering advanced features...")
        
        # Create a single row DataFrame with ALL original columns
        data = {
            'place_id': [place_id],
            'category': [category],
            'district': [district],
            'time_slot': [time_slot],
            'hour': [hour if hour is not None else (9 if time_slot == 'morning' else 14 if time_slot == 'afternoon' else 19)],
            'day_of_week': [day_of_week],
            'month': [month],
            'day': [1],  # Default day value
            'season': [season],
            'is_weekend': [is_weekend],
            'is_holiday': [is_holiday],
            'weather_condition': [weather_condition]
        }
        
        # Add tourist season if available
        if month in [10, 11, 4, 5]:
            data['tourist_season'] = ['Peak']
        elif month in [3, 9, 12]:
            data['tourist_season'] = ['Shoulder']
        else:
            data['tourist_season'] = ['Low']
        
        # Add festival period (simplified)
        data['festival_period'] = ['No']
        
        df = pd.DataFrame(data)
        
        # Engineer features
        df = self.engineer_features(df)
        
        # Get the original categorical features that the preprocessor expects
        categorical_features = ['category', 'district', 'time_slot', 'season', 'weather_condition', 'tourist_season', 'festival_period']
        
        # Create a DataFrame with only the features that the preprocessor expects
        # This should match exactly what was used during training
        X = df.copy()
        
        # Ensure all categorical features are present
        for cat_feature in categorical_features:
            if cat_feature not in X.columns:
                X[cat_feature] = 'Unknown'
        
        # Get all numerical features (everything except categorical and target)
        numerical_features = [col for col in X.columns if col not in categorical_features + ['crowdlevel', 'crowd_level_bin']]
        
        # Create the final feature DataFrame with the exact structure expected by the preprocessor
        final_features = X[categorical_features + numerical_features]
        
        # Transform features using the preprocessor
        X_processed = self.preprocessor.transform(final_features)
        
        # Make prediction
        prediction = self.model.predict(X_processed)[0]
        
        # Ensure prediction is within valid range
        prediction = max(0, min(100, prediction))
        
        return prediction
    
    def save_model(self):
        """Save the trained model and preprocessor"""
        try:
            model_data = {
                'model': self.model,
                'preprocessor': self.preprocessor,
                'feature_names': self.feature_names,
                'model_info': self.model_info
            }
            joblib.dump(model_data, self.model_path)
            logger.info("Enhanced model saved successfully!")
            return True
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            return False
    
    def load_model(self):
        """Load the trained model and preprocessor"""
        try:
            if not os.path.exists(self.model_path):
                logger.error(f"Model file not found: {self.model_path}")
                return False
            
            model_data = joblib.load(self.model_path)
            self.model = model_data['model']
            self.preprocessor = model_data['preprocessor']
            self.feature_names = model_data['feature_names']
            self.model_info = model_data.get('model_info', {})
            
            logger.info("Enhanced model loaded successfully!")
            return True
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
    
    def get_model_info(self):
        """Get information about the trained model"""
        return self.model_info

def main():
    """Main function for command line usage"""
    parser = argparse.ArgumentParser(description='Train enhanced crowd prediction model')
    parser.add_argument('--train_data', default='nepal_tourism_crowd_data.csv', help='Path to training CSV file')
    parser.add_argument('--model_output', default='crowd_prediction_model.joblib', help='Path to save the model')
    parser.add_argument('--force-retrain', action='store_true', help='Force retraining even if model exists')
    
    args = parser.parse_args()
    
    model = ImprovedCrowdPredictionModel(args.model_output)
    
    if os.path.exists(args.model_output) and not args.force_retrain:
        print(f"Model already exists at {args.model_output}")
        print("Use --force-retrain to retrain the model")
        return
    
    success = model.train_model(args.train_data)
    
    if success:
        print("🎉 Enhanced model training completed successfully!")
    else:
        print("❌ Model training failed!")
        exit(1)

if __name__ == "__main__":
    main() 