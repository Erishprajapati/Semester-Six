import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Load data and model
csv_path = 'balanced_crowd_data.csv'
model_path = 'backend/crowd_prediction_model.joblib'
df = pd.read_csv(csv_path)
model = joblib.load(model_path)

# Features used for prediction
feature_cols = [
    'place_id', 'category', 'district', 'hour', 'day_of_week', 'month',
    'season', 'is_weekend', 'weather_condition', 'time_slot'
]
X = pd.get_dummies(df[feature_cols].astype({'place_id': str}))

# Align columns with model (handle missing dummies)
if hasattr(model, 'feature_names_in_'):
    for col in model.feature_names_in_:
        if col not in X.columns:
            X[col] = 0
    X = X[model.feature_names_in_]

# Predict
preds = model.predict(X)
df['predicted_crowdlevel'] = preds

# Categorize predictions
high = (preds > 70).sum()
medium = ((preds >= 30) & (preds <= 70)).sum()
low = (preds < 30).sum()
print(f"Predicted High: {high}")
print(f"Predicted Medium: {medium}")
print(f"Predicted Low: {low}")

# Plot histogram
plt.figure(figsize=(8,5))
plt.hist(preds, bins=20, color='skyblue', edgecolor='black')
plt.title('Histogram of Predicted Crowd Levels')
plt.xlabel('Predicted Crowd Level (%)')
plt.ylabel('Count')
plt.axvline(70, color='red', linestyle='--', label='High/Medium threshold (70)')
plt.axvline(30, color='green', linestyle='--', label='Medium/Low threshold (30)')
plt.legend()
plt.tight_layout()
plt.show() 