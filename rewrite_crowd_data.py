import pandas as pd
import numpy as np

# Load the original CSV
df = pd.read_csv('enhanced_crowd_data.csv')

# Define logic for realistic crowd levels
def assign_crowdlevel(row):
    cat = str(row['category']).lower()
    slot = str(row['time_slot']).lower()
    base = 0
    # Markets and tourist squares
    if cat in ['market', 'historical', 'tourist', 'shopping', 'entertainment', 'cultural']:
        if slot == 'evening':
            base = np.random.uniform(80, 100)  # High
        elif slot == 'afternoon':
            base = np.random.uniform(55, 75)   # Medium
        else:
            base = np.random.uniform(20, 40)   # Low
    # Parks and resorts
    elif cat in ['park', 'resort', 'nature']:
        if slot == 'afternoon' or slot == 'evening':
            base = np.random.uniform(50, 70)   # Medium
        else:
            base = np.random.uniform(15, 35)   # Low
    # Temples (Religious)
    elif cat in ['temple', 'religious']:
        if slot == 'morning':
            base = np.random.uniform(75, 95)   # High
        elif slot == 'afternoon':
            base = np.random.uniform(45, 65)   # Medium
        else:
            base = np.random.uniform(15, 35)   # Low
    # Museums
    elif cat in ['museum']:
        if slot == 'afternoon':
            base = np.random.uniform(45, 65)   # Medium
        else:
            base = np.random.uniform(10, 30)   # Low
    else:
        # Default: medium in afternoon, low otherwise
        if slot == 'afternoon':
            base = np.random.uniform(40, 60)
        else:
            base = np.random.uniform(10, 30)
    return round(base, 1)

# Apply the logic
df['crowdlevel'] = df.apply(assign_crowdlevel, axis=1)

# Save back to the same file
df.to_csv('enhanced_crowd_data.csv', index=False)

print('✅ enhanced_crowd_data.csv rewritten with realistic high/medium/low crowd levels.') 