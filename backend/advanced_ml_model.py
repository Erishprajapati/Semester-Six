import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder, RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_regression, RFE
import joblib
import os
import warnings
import logging
from datetime import datetime, timedelta
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Try to import advanced libraries
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost not available. Install with: pip install xgboost")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("LightGBM not available. Install with: pip install lightgbm")

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

class AdvancedCrowdPredictionModel:
    def __init__(self, model_path='advanced_crowd_prediction_model.joblib'):
        self.model_path = model_path
        self.model = None
        self.preprocessor = None
        self.feature_names = None
        self.model_info = {}
        self.feature_importance = None
        self.validation_results = {}
        
    def advanced_data_cleaning(self, df):
        """Advanced data cleaning with statistical analysis"""
        logger.info("Starting advanced data cleaning and analysis...")
        
        initial_rows = len(df)
        
        # 1. Missing value analysis
        missing_data = df.isnull().sum()
        logger.info(f"Missing values per column:\n{missing_data[missing_data > 0]}")
        
        # Remove rows with missing values
        df = df.dropna()
        logger.info(f"Removed {initial_rows - len(df)} rows with missing values")
        
        # 2. Statistical outlier detection using Z-score and IQR
        if 'crowdlevel' in df.columns:
            # Z-score method
            z_scores = np.abs(stats.zscore(df['crowdlevel']))
            outliers_z = df[z_scores > 3]
            
            # IQR method
            Q1 = df['crowdlevel'].quantile(0.25)
            Q3 = df['crowdlevel'].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers_iqr = df[(df['crowdlevel'] < lower_bound) | (df['crowdlevel'] > upper_bound)]
            
            logger.info(f"Z-score outliers: {len(outliers_z)}")
            logger.info(f"IQR outliers: {len(outliers_iqr)}")
            
            # Remove extreme outliers (beyond 4 standard deviations)
            extreme_outliers = df[z_scores > 4]
            df = df[z_scores <= 4]
            logger.info(f"Removed {len(extreme_outliers)} extreme outliers")
        
        # 3. Data distribution analysis
        if 'crowdlevel' in df.columns:
            logger.info(f"Crowd level statistics:")
            logger.info(f"  Mean: {df['crowdlevel'].mean():.2f}")
            logger.info(f"  Median: {df['crowdlevel'].median():.2f}")
            logger.info(f"  Std: {df['crowdlevel'].std():.2f}")
            logger.info(f"  Min: {df['crowdlevel'].min():.2f}")
            logger.info(f"  Max: {df['crowdlevel'].max():.2f}")
        
        # 4. Remove duplicates
        initial_rows = len(df)
        df = df.drop_duplicates()
        logger.info(f"Removed {initial_rows - len(df)} duplicate rows")
        
        # 5. Validate crowd levels
        invalid_crowd = df[(df['crowdlevel'] < 0) | (df['crowdlevel'] > 100)]
        if len(invalid_crowd) > 0:
            df = df[(df['crowdlevel'] >= 0) & (df['crowdlevel'] <= 100)]
            logger.info(f"Removed {len(invalid_crowd)} rows with invalid crowd levels")
        
        return df
    
    def advanced_feature_engineering(self, df):
        """Advanced feature engineering with domain knowledge"""
        logger.info("Engineering advanced features with domain knowledge...")
        
        # 1. Cyclical encoding for time-based features
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        # 2. Advanced time-based features
        df['is_busy_day'] = df['is_weekend'] * df['is_holiday']
        df['is_peak_hour'] = ((df['hour'] >= 9) & (df['hour'] <= 11)) | ((df['hour'] >= 17) & (df['hour'] <= 19))
        df['is_off_peak'] = (df['hour'] >= 22) | (df['hour'] <= 6)
        df['is_rush_hour'] = ((df['hour'] >= 7) & (df['hour'] <= 9)) | ((df['hour'] >= 17) & (df['hour'] <= 19))
        
        # 3. Tourist season features with more granularity
        if 'tourist_season' in df.columns:
            df['is_peak_season'] = (df['tourist_season'] == 'Peak').astype(int)
            df['is_shoulder_season'] = (df['tourist_season'] == 'Shoulder').astype(int)
            df['is_low_season'] = (df['tourist_season'] == 'Low').astype(int)
        else:
            # Enhanced tourist season logic
            df['is_peak_season'] = df['month'].isin([10, 11, 4, 5]).astype(int)
            df['is_shoulder_season'] = df['month'].isin([3, 9, 12]).astype(int)
            df['is_low_season'] = df['month'].isin([6, 7, 8]).astype(int)
        
        # 4. Festival period features
        if 'festival_period' in df.columns:
            df['is_festival'] = (df['festival_period'] == 'Yes').astype(int)
        else:
            df['is_festival'] = 0
        
        # 5. Category-specific features with more granularity
        df['is_religious'] = (df['category'] == 'Religious').astype(int)
        df['is_historical'] = (df['category'] == 'Historical').astype(int)
        df['is_nature'] = (df['category'].isin(['Nature', 'Natural', 'Park'])).astype(int)
        df['is_market'] = (df['category'] == 'Market').astype(int)
        df['is_cultural'] = (df['category'] == 'Cultural').astype(int)
        df['is_museum'] = (df['category'] == 'Museum').astype(int)
        df['is_monument'] = (df['category'] == 'Monument').astype(int)
        df['is_viewpoint'] = (df['category'] == 'Viewpoint').astype(int)
        
        # 6. District features
        df['is_kathmandu'] = (df['district'] == 'Kathmandu').astype(int)
        df['is_lalitpur'] = (df['district'] == 'Lalitpur').astype(int)
        df['is_bhaktapur'] = (df['district'] == 'Bhaktapur').astype(int)
        
        # 7. Enhanced weather severity mapping
        weather_severity = {
            'Sunny': 1,
            'Cloudy': 2,
            'Foggy': 3,
            'Rainy': 4,
            'Stormy': 5
        }
        df['weather_severity'] = df['weather_condition'].map(weather_severity)
        
        # 8. Season features
        df['is_spring'] = (df['season'] == 'Spring').astype(int)
        df['is_summer'] = (df['season'] == 'Summer').astype(int)
        df['is_autumn'] = (df['season'] == 'Autumn').astype(int)
        df['is_winter'] = (df['season'] == 'Winter').astype(int)
        
        # 9. Time slot features
        df['is_morning'] = (df['time_slot'] == 'morning').astype(int)
        df['is_afternoon'] = (df['time_slot'] == 'afternoon').astype(int)
        df['is_evening'] = (df['time_slot'] == 'evening').astype(int)
        
        # 10. Advanced interaction features
        df['weekend_morning'] = df['is_weekend'] * df['is_morning']
        df['weekend_evening'] = df['is_weekend'] * df['is_evening']
        df['peak_season_weekend'] = df['is_peak_season'] * df['is_weekend']
        df['religious_weekend'] = df['is_religious'] * df['is_weekend']
        df['market_peak_hour'] = df['is_market'] * df['is_peak_hour']
        df['nature_sunny'] = df['is_nature'] * (df['weather_condition'] == 'Sunny').astype(int)
        df['religious_festival'] = df['is_religious'] * df['is_festival']
        df['market_holiday'] = df['is_market'] * df['is_holiday']
        
        # 11. Crowd level bins for feature engineering
        if 'crowdlevel' in df.columns:
            df['crowd_level_bin'] = pd.cut(df['crowdlevel'], bins=[0, 30, 70, 100], labels=['Low', 'Medium', 'High'])
            df['is_high_crowd'] = (df['crowdlevel'] > 70).astype(int)
            df['is_low_crowd'] = (df['crowdlevel'] < 30).astype(int)
        
        # 12. Place-specific features (if place_id is available)
        if 'place_id' in df.columns:
            # Calculate average crowd level per place
            place_avg_crowd = df.groupby('place_id')['crowdlevel'].mean().reset_index()
            place_avg_crowd.columns = ['place_id', 'place_avg_crowd']
            df = df.merge(place_avg_crowd, on='place_id', how='left')
            
            # Calculate crowd level deviation from place average
            df['crowd_deviation'] = df['crowdlevel'] - df['place_avg_crowd']
        
        logger.info(f"Engineered {len([col for col in df.columns if col not in ['place_id', 'category', 'district', 'time_slot', 'hour', 'day_of_week', 'month', 'day', 'season', 'is_weekend', 'is_holiday', 'weather_condition', 'crowdlevel', 'tourist_season', 'festival_period']])} total features")
        
        return df
    
    def prepare_advanced_data(self, df):
        """Prepare data with advanced preprocessing"""
        logger.info("Preparing data with advanced preprocessing...")
        
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
        
        # Create advanced preprocessing pipeline
        numerical_transformer = RobustScaler()  # More robust to outliers
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
    
    def train_advanced_model(self, csv_file='nepal_tourism_crowd_data.csv'):
        """Train advanced model with hyperparameter tuning and ensemble methods"""
        logger.info("🚀 Starting advanced crowd prediction model training...")
        
        try:
            # Load data
            logger.info(f"Loading data from {csv_file}...")
            df = pd.read_csv(csv_file)
            
            # Advanced data cleaning
            df = self.advanced_data_cleaning(df)
            
            # Advanced feature engineering
            df = self.advanced_feature_engineering(df)
            
            # Prepare data
            X, y = self.prepare_advanced_data(df)
            
            # Time-series aware split (if data has temporal order)
            if 'timestamp' in df.columns:
                # Sort by timestamp and use time-series split
                df_sorted = df.sort_values('timestamp')
                split_idx = int(len(df_sorted) * 0.8)
                X_train, X_test = X[:split_idx], X[split_idx:]
                y_train, y_test = y[:split_idx], y[split_idx:]
                logger.info("Using time-series aware split")
            else:
                # Regular split
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=pd.cut(y, bins=5))
                logger.info("Using stratified random split")
            
            # 1. Define advanced models with hyperparameter grids
            models_and_params = {
                'RandomForest': {
                    'model': RandomForestRegressor(random_state=42),
                    'params': {
                        'n_estimators': [100, 200, 300],
                        'max_depth': [10, 20, None],
                        'min_samples_split': [2, 5, 10],
                        'min_samples_leaf': [1, 2, 4]
                    }
                },
                'GradientBoosting': {
                    'model': GradientBoostingRegressor(random_state=42),
                    'params': {
                        'n_estimators': [100, 200],
                        'learning_rate': [0.05, 0.1, 0.2],
                        'max_depth': [3, 5, 7],
                        'subsample': [0.8, 0.9, 1.0]
                    }
                }
            }
            
            if XGBOOST_AVAILABLE:
                models_and_params['XGBoost'] = {
                    'model': xgb.XGBRegressor(random_state=42),
                    'params': {
                        'n_estimators': [100, 200],
                        'learning_rate': [0.05, 0.1, 0.2],
                        'max_depth': [3, 5, 7],
                        'subsample': [0.8, 0.9, 1.0],
                        'colsample_bytree': [0.8, 0.9, 1.0]
                    }
                }
            
            if LIGHTGBM_AVAILABLE:
                models_and_params['LightGBM'] = {
                    'model': lgb.LGBMRegressor(random_state=42, verbose=-1),
                    'params': {
                        'n_estimators': [100, 200],
                        'learning_rate': [0.05, 0.1, 0.2],
                        'max_depth': [3, 5, 7],
                        'subsample': [0.8, 0.9, 1.0],
                        'colsample_bytree': [0.8, 0.9, 1.0]
                    }
                }
            
            # 2. Hyperparameter tuning with cross-validation
            logger.info("Performing hyperparameter tuning...")
            best_models = {}
            
            for name, config in models_and_params.items():
                logger.info(f"Tuning {name}...")
                
                # Use TimeSeriesSplit for time-series data
                if 'timestamp' in df.columns:
                    cv = TimeSeriesSplit(n_splits=5)
                else:
                    cv = 5
                
                grid_search = GridSearchCV(
                    config['model'],
                    config['params'],
                    cv=cv,
                    scoring='r2',
                    n_jobs=-1,
                    verbose=0
                )
                
                grid_search.fit(X_train, y_train)
                best_models[name] = grid_search.best_estimator_
                
                logger.info(f"{name} best params: {grid_search.best_params_}")
                logger.info(f"{name} best CV score: {grid_search.best_score_:.4f}")
            
            # 3. Ensemble model creation
            logger.info("Creating ensemble model...")
            
            # Create voting regressor with best models
            estimators = [(name, model) for name, model in best_models.items()]
            self.model = VotingRegressor(estimators=estimators, n_jobs=-1)
            
            # Train ensemble
            self.model.fit(X_train, y_train)
            
            # 4. Comprehensive evaluation
            logger.info("Evaluating model performance...")
            
            # Train predictions
            y_train_pred = self.model.predict(X_train)
            train_mse = mean_squared_error(y_train, y_train_pred)
            train_mae = mean_absolute_error(y_train, y_train_pred)
            train_r2 = r2_score(y_train, y_train_pred)
            train_mape = mean_absolute_percentage_error(y_train, y_train_pred)
            
            # Test predictions
            y_test_pred = self.model.predict(X_test)
            test_mse = mean_squared_error(y_test, y_test_pred)
            test_mae = mean_absolute_error(y_test, y_test_pred)
            test_r2 = r2_score(y_test, y_test_pred)
            test_mape = mean_absolute_percentage_error(y_test, y_test_pred)
            
            # Cross-validation scores
            cv_scores = cross_val_score(self.model, X_train, y_train, cv=5, scoring='r2')
            
            logger.info(f"\n📊 Model Performance:")
            logger.info(f"Training - MSE: {train_mse:.2f}, MAE: {train_mae:.2f}, R²: {train_r2:.4f}, MAPE: {train_mape:.4f}")
            logger.info(f"Testing  - MSE: {test_mse:.2f}, MAE: {test_mae:.2f}, R²: {test_r2:.4f}, MAPE: {test_mape:.4f}")
            logger.info(f"CV R²: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
            
            # 5. Feature importance analysis
            self.analyze_feature_importance(X_train, y_train)
            
            # 6. Save comprehensive model info
            self.model_info = {
                'model_type': 'Ensemble',
                'ensemble_models': list(best_models.keys()),
                'train_r2': train_r2,
                'test_r2': test_r2,
                'train_mae': train_mae,
                'test_mae': test_mae,
                'train_mse': train_mse,
                'test_mse': test_mse,
                'train_mape': train_mape,
                'test_mape': test_mape,
                'cv_r2_mean': cv_scores.mean(),
                'cv_r2_std': cv_scores.std(),
                'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'data_source': csv_file,
                'total_samples': len(df),
                'total_features': X.shape[1],
                'shap_available': SHAP_AVAILABLE
            }
            
            # 7. Save the model
            self.save_model()
            
            # 8. Generate validation report
            self.generate_validation_report(y_test, y_test_pred)
            
            return True
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def analyze_feature_importance(self, X_train, y_train):
        """Analyze feature importance using multiple methods"""
        logger.info("Analyzing feature importance...")
        
        # Get feature importance from the best individual model
        best_individual_model = None
        best_score = -np.inf
        
        for name, model in self.model.estimators_:
            if hasattr(model, 'feature_importances_'):
                score = model.score(X_train, y_train)
                if score > best_score:
                    best_score = score
                    best_individual_model = model
        
        if best_individual_model and hasattr(best_individual_model, 'feature_importances_'):
            feature_importance = best_individual_model.feature_importances_
            feature_importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': feature_importance
            }).sort_values('importance', ascending=False)
            
            self.feature_importance = feature_importance_df
            
            logger.info(f"\n🏆 Top 15 Most Important Features:")
            for i, row in feature_importance_df.head(15).iterrows():
                logger.info(f"  📈 {row['feature']}: {row['importance']:.4f}")
    
    def generate_validation_report(self, y_true, y_pred):
        """Generate comprehensive validation report"""
        logger.info("Generating validation report...")
        
        # Calculate residuals
        residuals = y_true - y_pred
        
        # Statistical analysis
        self.validation_results = {
            'residuals_mean': residuals.mean(),
            'residuals_std': residuals.std(),
            'residuals_skew': stats.skew(residuals),
            'residuals_kurtosis': stats.kurtosis(residuals),
            'prediction_bias': y_pred.mean() - y_true.mean(),
            'prediction_variance': y_pred.var() / y_true.var() if y_true.var() > 0 else 0
        }
        
        logger.info(f"\n📋 Validation Report:")
        logger.info(f"  Residuals - Mean: {self.validation_results['residuals_mean']:.3f}, Std: {self.validation_results['residuals_std']:.3f}")
        logger.info(f"  Residuals - Skewness: {self.validation_results['residuals_skew']:.3f}, Kurtosis: {self.validation_results['residuals_kurtosis']:.3f}")
        logger.info(f"  Prediction Bias: {self.validation_results['prediction_bias']:.3f}")
        logger.info(f"  Prediction Variance Ratio: {self.validation_results['prediction_variance']:.3f}")
    
    def predict(self, place_id, category, district, time_slot, day_of_week, month, season, is_weekend, is_holiday, weather_condition, hour=None):
        """Make predictions with advanced feature engineering"""
        if self.model is None:
            raise ValueError("Model not loaded. Please train or load the model first.")
        
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
        
        # Add tourist season
        if month in [10, 11, 4, 5]:
            data['tourist_season'] = ['Peak']
        elif month in [3, 9, 12]:
            data['tourist_season'] = ['Shoulder']
        else:
            data['tourist_season'] = ['Low']
        
        # Add festival period
        data['festival_period'] = ['No']
        
        df = pd.DataFrame(data)
        
        # Engineer features
        df = self.advanced_feature_engineering(df)
        
        # Get categorical features
        categorical_features = ['category', 'district', 'time_slot', 'season', 'weather_condition', 'tourist_season', 'festival_period']
        
        # Create feature DataFrame
        X = df.copy()
        
        # Ensure all categorical features are present
        for cat_feature in categorical_features:
            if cat_feature not in X.columns:
                X[cat_feature] = 'Unknown'
        
        # Get numerical features
        numerical_features = [col for col in X.columns if col not in categorical_features + ['crowdlevel', 'crowd_level_bin']]
        
        # Create final feature DataFrame
        final_features = X[categorical_features + numerical_features]
        
        # Transform features
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
                'model_info': self.model_info,
                'feature_importance': self.feature_importance,
                'validation_results': self.validation_results
            }
            joblib.dump(model_data, self.model_path)
            logger.info("Advanced model saved successfully!")
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
            self.feature_importance = model_data.get('feature_importance', None)
            self.validation_results = model_data.get('validation_results', {})
            
            logger.info("Advanced model loaded successfully!")
            return True
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
    
    def get_model_info(self):
        """Get comprehensive information about the trained model"""
        return self.model_info

def main():
    """Main function for command line usage"""
    parser = argparse.ArgumentParser(description='Train advanced crowd prediction model')
    parser.add_argument('--train_data', default='nepal_tourism_crowd_data.csv', help='Path to training CSV file')
    parser.add_argument('--model_output', default='advanced_crowd_prediction_model.joblib', help='Path to save the model')
    parser.add_argument('--force-retrain', action='store_true', help='Force retraining even if model exists')
    
    args = parser.parse_args()
    
    model = AdvancedCrowdPredictionModel(args.model_output)
    
    if os.path.exists(args.model_output) and not args.force_retrain:
        print(f"Model already exists at {args.model_output}")
        print("Use --force-retrain to retrain the model")
        return
    
    success = model.train_advanced_model(args.train_data)
    
    if success:
        print("🎉 Advanced model training completed successfully!")
    else:
        print("❌ Model training failed!")
        exit(1)

if __name__ == "__main__":
    main() 