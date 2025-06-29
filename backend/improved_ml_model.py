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

warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ImprovedCrowdPredictionModel:
    def __init__(self):
        self.model = None
        self.preprocessor = None
        self.label_encoders = {}
        self.is_trained = False
        self.feature_importance = None
        self.model_performance = {}
        self.best_model_name = None
        self.shap_explainer = None
        
    def clean_data(self, df):
        """Clean the dataset by removing missing values and outliers"""
        logger.info("Starting data cleaning...")
        
        # Remove missing values
        initial_rows = len(df)
        df = df.dropna()
        logger.info(f"Removed {initial_rows - len(df)} rows with missing values")
        
        # Remove invalid crowd levels
        df = df[df['crowdlevel'] >= 0]
        df = df[df['crowdlevel'] <= 100]
        logger.info(f"Removed invalid crowd levels. Remaining rows: {len(df)}")
        
        # Remove outliers using IQR method for crowdlevel
        Q1 = df['crowdlevel'].quantile(0.25)
        Q3 = df['crowdlevel'].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        df_clean = df[(df['crowdlevel'] >= lower_bound) & (df['crowdlevel'] <= upper_bound)]
        logger.info(f"Removed {len(df) - len(df_clean)} outliers using IQR method")
        
        return df_clean
    
    def engineer_features(self, df):
        """Create advanced features for better prediction"""
        logger.info("Engineering advanced features...")
        
        # Cyclical encoding for time-based features
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        # Time slot features
        df['is_morning'] = (df['time_slot'] == 'morning').astype(int)
        df['is_afternoon'] = (df['time_slot'] == 'afternoon').astype(int)
        df['is_evening'] = (df['time_slot'] == 'evening').astype(int)
        
        # Interaction features
        df['weekend_holiday'] = df['is_weekend'] * df['is_holiday']
        df['is_busy_day'] = df['is_weekend'] | df['is_holiday']
        
        # Seasonal features
        df['tourist_season'] = ((df['month'] >= 6) & (df['month'] <= 8)).astype(int)
        df['off_season'] = ((df['month'] >= 12) | (df['month'] <= 2)).astype(int)
        df['shoulder_season'] = ((df['month'] >= 3) & (df['month'] <= 5)) | ((df['month'] >= 9) & (df['month'] <= 11))
        df['shoulder_season'] = df['shoulder_season'].astype(int)
        
        # Weather severity mapping
        weather_severity = {
            'Sunny': 1,
            'Cloudy': 2,
            'Foggy': 3,
            'Rainy': 4
        }
        df['weather_severity'] = df['weather_condition'].map(weather_severity)
        
        # Category-specific features
        df['is_religious'] = (df['category'].str.contains('Temple|Religious', case=False, na=False)).astype(int)
        df['is_nature'] = (df['category'].str.contains('Nature|Park', case=False, na=False)).astype(int)
        df['is_historical'] = (df['category'].str.contains('Historical|Heritage', case=False, na=False)).astype(int)
        df['is_entertainment'] = (df['category'].str.contains('Entertainment|Shopping', case=False, na=False)).astype(int)
        
        # District features
        df['is_kathmandu'] = (df['district'] == 'Kathmandu').astype(int)
        df['is_lalitpur'] = (df['district'] == 'Lalitpur').astype(int)
        df['is_bhaktapur'] = (df['district'] == 'Bhaktapur').astype(int)
        
        # Peak hour features
        df['is_peak_hour'] = ((df['hour'] >= 9) & (df['hour'] <= 17)).astype(int)
        df['is_early_morning'] = ((df['hour'] >= 6) & (df['hour'] <= 8)).astype(int)
        df['is_late_evening'] = ((df['hour'] >= 19) & (df['hour'] <= 21)).astype(int)
        
        logger.info(f"Engineered {len(df.columns)} total features")
        return df
    
    def prepare_data(self, csv_file):
        """Prepare data with advanced cleaning and feature engineering"""
        logger.info(f"Loading data from {csv_file}...")
        df = pd.read_csv(csv_file)
        
        # Clean data
        df = self.clean_data(df)
        
        # Engineer features
        df = self.engineer_features(df)
        
        # Separate categorical and numerical features
        categorical_features = ['category', 'district', 'time_slot', 'season', 'weather_condition']
        numerical_features = [
            'place_id', 'hour', 'day_of_week', 'month', 'is_weekend', 'is_holiday',
            'hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'month_sin', 'month_cos',
            'is_morning', 'is_afternoon', 'is_evening', 'weekend_holiday', 'is_busy_day',
            'tourist_season', 'off_season', 'shoulder_season', 'weather_severity',
            'is_religious', 'is_nature', 'is_historical', 'is_entertainment',
            'is_kathmandu', 'is_lalitpur', 'is_bhaktapur', 'is_peak_hour',
            'is_early_morning', 'is_late_evening'
        ]
        
        # Create target variable
        y = df['crowdlevel']
        
        # Create feature matrix
        X = df[categorical_features + numerical_features]
        
        logger.info(f"Prepared data: {X.shape[0]} samples, {X.shape[1]} features")
        return X, y, categorical_features, numerical_features
    
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
        """Select the best model using 5-fold cross-validation"""
        logger.info("Selecting best model using 5-fold cross-validation...")
        
        # Define models to test
        models = {
            'RandomForest': RandomForestRegressor(
                n_estimators=100, 
                max_depth=10, 
                random_state=42,
                n_jobs=-1
            ),
            'GradientBoosting': GradientBoostingRegressor(
                n_estimators=100, 
                max_depth=6, 
                random_state=42
            )
        }
        
        # Add XGBoost if available
        if XGBOOST_AVAILABLE:
            models['XGBoost'] = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=6,
                random_state=42,
                n_jobs=-1
            )
        
        # Define preprocessing
        categorical_features = ['category', 'district', 'time_slot', 'season', 'weather_condition']
        numerical_features = [
            'place_id', 'hour', 'day_of_week', 'month', 'is_weekend', 'is_holiday',
            'hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'month_sin', 'month_cos',
            'is_morning', 'is_afternoon', 'is_evening', 'weekend_holiday', 'is_busy_day',
            'tourist_season', 'off_season', 'shoulder_season', 'weather_severity',
            'is_religious', 'is_nature', 'is_historical', 'is_entertainment',
            'is_kathmandu', 'is_lalitpur', 'is_bhaktapur', 'is_peak_hour',
            'is_early_morning', 'is_late_evening'
        ]
        
        preprocessor = self.create_preprocessor(categorical_features, numerical_features)
        
        best_score = -np.inf
        best_model_name = None
        best_model = None
        
        # 5-fold cross-validation
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        
        for name, model in models.items():
            # Create pipeline
            pipeline = Pipeline([
                ('preprocessor', preprocessor),
                ('regressor', model)
            ])
            
            # Cross-validation
            scores = cross_val_score(pipeline, X, y, cv=cv, scoring='r2')
            mean_score = scores.mean()
            std_score = scores.std()
            
            logger.info(f"{name}: R² = {mean_score:.4f} (+/- {std_score * 2:.4f})")
            
            if mean_score > best_score:
                best_score = mean_score
                best_model_name = name
                best_model = pipeline
        
        logger.info(f"\nBest model: {best_model_name} with R² = {best_score:.4f}")
        return best_model, best_model_name
    
    def train(self, csv_file):
        """Train the improved model with advanced features"""
        logger.info("Starting enhanced model training...")
        
        # Prepare data
        X, y, categorical_features, numerical_features = self.prepare_data(csv_file)
        
        # Select best model
        self.model, self.best_model_name = self.select_best_model(X, y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=pd.cut(y, bins=10, labels=False)
        )
        
        logger.info(f"Training {self.best_model_name}...")
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
            'model_name': self.best_model_name
        }
        
        logger.info(f"\nModel Performance:")
        logger.info(f"Mean Squared Error: {mse:.2f}")
        logger.info(f"Mean Absolute Error: {mae:.2f}")
        logger.info(f"R² Score: {r2:.4f}")
        
        # Get feature importance if available
        if hasattr(self.model.named_steps['regressor'], 'feature_importances_'):
            feature_names = self.model.named_steps['preprocessor'].get_feature_names_out()
            self.feature_importance = pd.DataFrame({
                'feature': feature_names,
                'importance': self.model.named_steps['regressor'].feature_importances_
            }).sort_values('importance', ascending=False)
            
            logger.info(f"\nTop 10 Most Important Features:")
            for idx, row in self.feature_importance.head(10).iterrows():
                logger.info(f"  {row['feature']}: {row['importance']:.4f}")
        
        # Create SHAP explainer if available
        if SHAP_AVAILABLE:
            try:
                logger.info("Creating SHAP explainer...")
                self.shap_explainer = shap.Explainer(self.model, X_train[:100])  # Use subset for speed
                logger.info("SHAP explainer created successfully")
            except Exception as e:
                logger.warning(f"Could not create SHAP explainer: {e}")
        
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
        df = self.engineer_features(df)
        
        # Make prediction
        prediction = self.model.predict(df)[0]
        
        # Ensure prediction is within bounds
        prediction = max(0, min(100, prediction))
        
        return round(prediction, 1)
    
    def explain_prediction(self, place_id, category, district, time_slot, day_of_week, 
                          month, season, is_weekend, is_holiday, weather_condition, hour=None):
        """Explain prediction using SHAP values"""
        if not self.is_trained or self.shap_explainer is None:
            return None
        
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
        df = self.engineer_features(df)
        
        # Get SHAP values
        shap_values = self.shap_explainer(df)
        
        return shap_values
    
    def save_model(self):
        """Save the trained model"""
        model_data = {
            'model': self.model,
            'is_trained': self.is_trained,
            'model_performance': self.model_performance,
            'feature_importance': self.feature_importance,
            'best_model_name': self.best_model_name,
            'shap_explainer': self.shap_explainer
        }
        joblib.dump(model_data, 'improved_crowd_prediction_model.pkl')
        logger.info("Enhanced model saved successfully!")
    
    def load_model(self):
        """Load a trained model"""
        if os.path.exists('improved_crowd_prediction_model.pkl'):
            model_data = joblib.load('improved_crowd_prediction_model.pkl')
            self.model = model_data['model']
            self.is_trained = model_data['is_trained']
            self.model_performance = model_data.get('model_performance', {})
            self.feature_importance = model_data.get('feature_importance', None)
            self.best_model_name = model_data.get('best_model_name', 'Unknown')
            self.shap_explainer = model_data.get('shap_explainer', None)
            logger.info("Enhanced model loaded successfully!")
            return True
        else:
            logger.warning("No saved enhanced model found!")
            return False
    
    def get_model_info(self):
        """Get information about the trained model"""
        if not self.is_trained:
            return "Model not trained"
        
        info = {
            'model_name': self.best_model_name or self.model_performance.get('model_name', 'Unknown'),
            'r2_score': self.model_performance.get('r2', 0),
            'mae': self.model_performance.get('mae', 0),
            'mse': self.model_performance.get('mse', 0),
            'shap_available': self.shap_explainer is not None
        }
        
        return info

def main():
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(description='Train Enhanced Crowd Prediction Model')
    parser.add_argument('--train_data', type=str, default='improved_crowd_training_data.csv',
                       help='Path to training data CSV file')
    parser.add_argument('--model_output', type=str, default='improved_crowd_prediction_model.pkl',
                       help='Path to save the trained model')
    parser.add_argument('--force_retrain', action='store_true',
                       help='Force retraining even if model exists')
    
    args = parser.parse_args()
    
    # Check if training data exists
    if not os.path.exists(args.train_data):
        logger.error(f'Training data file {args.train_data} not found!')
        return
    
    try:
        # Initialize enhanced model
        model = ImprovedCrowdPredictionModel()
        
        # Check if model already exists and user doesn't want to retrain
        if not args.force_retrain and model.load_model():
            logger.info('Model already exists. Use --force_retrain to retrain.')
            info = model.get_model_info()
            logger.info(f"Current model: {info['model_name']}")
            logger.info(f"R² Score: {info['r2_score']:.4f}")
            logger.info(f"MAE: {info['mae']:.2f}")
            return
        
        logger.info('Training enhanced model...')
        mse, mae, r2 = model.train(args.train_data)
        
        logger.info(f'\nEnhanced model training completed successfully!')
        logger.info(f'Model: {model.best_model_name}')
        logger.info(f'Mean Squared Error: {mse:.2f}')
        logger.info(f'Mean Absolute Error: {mae:.2f}')
        logger.info(f'R² Score: {r2:.4f}')
        
        # Test predictions
        logger.info('\nTesting predictions...')
        from datetime import datetime
        
        # Test case 1: Temple on weekend morning
        test_prediction1 = model.predict(
            place_id=1,
            category='Temple',
            district='Kathmandu',
            time_slot='morning',
            day_of_week=6,  # Sunday
            month=datetime.now().month,
            season='Spring',
            is_weekend=1,
            is_holiday=0,
            weather_condition='Sunny'
        )
        logger.info(f'Test 1 (Temple, Sunday morning, sunny): {test_prediction1}% crowd level')
        
        # Test case 2: Market on weekday afternoon
        test_prediction2 = model.predict(
            place_id=2,
            category='Market',
            district='Kathmandu',
            time_slot='afternoon',
            day_of_week=2,  # Wednesday
            month=datetime.now().month,
            season='Spring',
            is_weekend=0,
            is_holiday=0,
            weather_condition='Cloudy'
        )
        logger.info(f'Test 2 (Market, Wednesday afternoon, cloudy): {test_prediction2}% crowd level')
        
        # Test case 3: Park on weekend evening
        test_prediction3 = model.predict(
            place_id=3,
            category='Park',
            district='Kathmandu',
            time_slot='evening',
            day_of_week=5,  # Saturday
            month=datetime.now().month,
            season='Spring',
            is_weekend=1,
            is_holiday=0,
            weather_condition='Sunny'
        )
        logger.info(f'Test 3 (Park, Saturday evening, sunny): {test_prediction3}% crowd level')
        
        # Show feature importance if available
        if model.feature_importance is not None:
            logger.info('\nTop 5 Most Important Features:')
            for idx, row in model.feature_importance.head().iterrows():
                logger.info(f'  {row["feature"]}: {row["importance"]:.4f}')
        
    except Exception as e:
        logger.error(f'Error training enhanced model: {str(e)}')
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main() 