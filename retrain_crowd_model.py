import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib

# Load your data
csv_path = 'enhanced_crowd_data.csv'
df = pd.read_csv(csv_path)

# Features and target (add 'place_id' as a categorical feature)
feature_cols = [
    'place_id', 'category', 'district', 'hour', 'day_of_week', 'month',
    'season', 'is_weekend', 'weather_condition', 'time_slot'
]
target_col = 'crowdlevel'

# Encode categorical features (including place_id)
X = pd.get_dummies(df[feature_cols].astype({'place_id': str}))
y = df[target_col]

# Train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# Save the model
joblib.dump(model, 'backend/crowd_prediction_model.joblib')

print('✅ Model retrained with place_id as a feature and saved as backend/crowd_prediction_model.joblib') 