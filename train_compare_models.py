import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import matplotlib.pyplot as plt
import os

# Try to import XGBoost
try:
    from xgboost import XGBRegressor
    xgb_available = True
except ImportError:
    xgb_available = False
    print('XGBoost not installed. Only Random Forest will be used.')

# Load data
csv_file = 'realistic_crowd_training_data.csv'
df = pd.read_csv(csv_file)

# Encode categorical variables
categorical_columns = ['category', 'district', 'time_slot', 'season', 'weather_condition']
label_encoders = {}
for col in categorical_columns:
    le = LabelEncoder()
    df[f'{col}_encoded'] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

# Feature columns
feature_columns = [
    'place_id', 'category_encoded', 'district_encoded', 'time_slot_encoded',
    'day_of_week', 'month', 'season_encoded', 'is_weekend', 'is_holiday',
    'weather_condition_encoded'
]
X = df[feature_columns]
y = df['crowdlevel']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, shuffle=True, random_state=42)

# Train Random Forest
print('Training Random Forest...')
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

# Cross-validation
rf_cv_scores = cross_val_score(rf, X, y, cv=5, scoring='neg_mean_squared_error')

# Metrics
rf_mse = mean_squared_error(y_test, rf_pred)
rf_mae = mean_absolute_error(y_test, rf_pred)
rf_r2 = r2_score(y_test, rf_pred)
print(f'Random Forest - MSE: {rf_mse:.2f}, MAE: {rf_mae:.2f}, R²: {rf_r2:.2f}')
print(f'Random Forest - 5-fold CV RMSE: {np.mean(np.sqrt(-rf_cv_scores)):.2f}')

# Feature importance plot
plt.figure(figsize=(10, 5))
plt.barh(feature_columns, rf.feature_importances_)
plt.title('Random Forest Feature Importance')
plt.tight_layout()
plt.savefig('rf_feature_importance.png')
plt.close()

# Train XGBoost (if available)
if xgb_available:
    print('Training XGBoost...')
    xgb = XGBRegressor(n_estimators=100, random_state=42, verbosity=0)
    xgb.fit(X_train, y_train)
    xgb_pred = xgb.predict(X_test)
    xgb_cv_scores = cross_val_score(xgb, X, y, cv=5, scoring='neg_mean_squared_error')
    xgb_mse = mean_squared_error(y_test, xgb_pred)
    xgb_mae = mean_absolute_error(y_test, xgb_pred)
    xgb_r2 = r2_score(y_test, xgb_pred)
    print(f'XGBoost - MSE: {xgb_mse:.2f}, MAE: {xgb_mae:.2f}, R²: {xgb_r2:.2f}')
    print(f'XGBoost - 5-fold CV RMSE: {np.mean(np.sqrt(-xgb_cv_scores)):.2f}')
    plt.figure(figsize=(10, 5))
    plt.barh(feature_columns, xgb.feature_importances_)
    plt.title('XGBoost Feature Importance')
    plt.tight_layout()
    plt.savefig('xgb_feature_importance.png')
    plt.close()
else:
    xgb = None
    xgb_r2 = -np.inf

# Save the best model
if xgb_available and xgb_r2 > rf_r2:
    best_model = xgb
    best_name = 'XGBoost'
else:
    best_model = rf
    best_name = 'Random Forest'

joblib.dump({'model': best_model, 'label_encoders': label_encoders}, 'crowd_prediction_model.pkl')
print(f'Saved best model: {best_name} as crowd_prediction_model.pkl')

# Plot error distribution
plt.figure(figsize=(8, 4))
plt.hist(y_test - rf_pred, bins=30, alpha=0.7, label='Random Forest')
if xgb_available:
    plt.hist(y_test - xgb_pred, bins=30, alpha=0.7, label='XGBoost')
plt.title('Prediction Error Distribution')
plt.xlabel('Error')
plt.ylabel('Frequency')
plt.legend()
plt.tight_layout()
plt.savefig('model_error_distribution.png')
plt.close()

print('Training and evaluation complete. Plots saved as rf_feature_importance.png, xgb_feature_importance.png, and model_error_distribution.png.') 