import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

# Load your data
csv_path = 'balanced_crowd_data.csv'
df = pd.read_csv(csv_path)

# Features and target (add 'place_id' as a categorical feature)
feature_cols = [
    'place_id', 'category', 'district', 'hour', 'day_of_week', 'month',
    'season', 'is_weekend', 'weather_condition', 'time_slot'
]
target_col = 'crowd_category'

# Encode categorical features (including place_id)
X = pd.get_dummies(df[feature_cols].astype({'place_id': str}))
y = df[target_col]

# Split for validation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the model
joblib.dump(model, 'backend/crowd_prediction_model.joblib')

# Print label distribution and a sample of predictions
print('Label distribution in training set:')
print(y_train.value_counts())
y_pred = model.predict(X_test)
print('\nSample predictions:')
print(pd.Series(y_pred).value_counts())

print('✅ Model retrained as classifier and saved as backend/crowd_prediction_model.joblib') 