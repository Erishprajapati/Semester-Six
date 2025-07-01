import pandas as pd
import sys
import os

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.improved_ml_model import ImprovedCrowdPredictionModel

# 1. Load ONLY the realistic enhanced crowd data
csv_file = 'enhanced_crowd_data.csv'
df = pd.read_csv(csv_file)

print(f"[INFO] Loaded {len(df)} rows from {csv_file}")

# 2. Drop 'place_id' from features for training
if 'place_id' in df.columns:
    df = df.drop(columns=['place_id'])

# 3. Save (optional, just to ensure format)
df.to_csv('enhanced_crowd_data_noid.csv', index=False)

# 4. Train model (pass the new CSV without place_id)
model = ImprovedCrowdPredictionModel('crowd_prediction_model.joblib')
model.train_model('enhanced_crowd_data_noid.csv')

# 5. Log metrics
info = model.get_model_info()
print(f"\nMean Squared Error: {info.get('mse', 0):.2f}")
print(f"R² Score: {info.get('r2_score', 0):.4f}")

# 6. Feature importance (top 5)
if hasattr(model.model, 'feature_importances_'):
    importances = model.model.feature_importances_
    features = model.feature_names
    fi_df = pd.DataFrame({'feature': features, 'importance': importances})
    fi_df = fi_df.sort_values('importance', ascending=False)
    print("\nTop 5 Feature Importances:")
    print(fi_df.head(5))

# 7. Test predictions (optional, can add more cases)
test_cases = [
    {'name': 'Park, Saturday morning, sunny', 'params': {'category': 'Park', 'district': 'Lalitpur', 'time_slot': 'morning', 'day_of_week': 5, 'month': 4, 'season': 'Spring', 'is_weekend': 1, 'is_holiday': 0, 'weather_condition': 'Sunny', 'hour': 9}},
    {'name': 'Temple, Sunday morning, rainy', 'params': {'category': 'Religious', 'district': 'Kathmandu', 'time_slot': 'morning', 'day_of_week': 6, 'month': 11, 'season': 'Autumn', 'is_weekend': 1, 'is_holiday': 0, 'weather_condition': 'Rainy', 'hour': 9}},
    {'name': 'Museum, Wednesday afternoon, cloudy', 'params': {'category': 'Museum', 'district': 'Kathmandu', 'time_slot': 'afternoon', 'day_of_week': 2, 'month': 7, 'season': 'Summer', 'is_weekend': 0, 'is_holiday': 0, 'weather_condition': 'Cloudy', 'hour': 14}},
]
print("\nTest predictions:")
for case in test_cases:
    pred = model.predict(**case['params'])
    print(f"{case['name']}: {pred:.1f}% crowd level")

print("\n✅ Model trained ONLY on your realistic enhanced_crowd_data.csv WITHOUT place_id!") 