import pandas as pd
import numpy as np

csv_path = 'enhanced_crowd_data.csv'
df = pd.read_csv(csv_path)

# Define bins
def get_category(crowdlevel):
    if crowdlevel > 70:
        return 'High'
    elif crowdlevel >= 30:
        return 'Medium'
    else:
        return 'Low'

df['crowd_category'] = df['crowdlevel'].apply(get_category)

change_count = 0
for district in df['district'].unique():
    for time_slot in df['time_slot'].unique():
        subset = df[(df['district'] == district) & (df['time_slot'] == time_slot)]
        counts = subset['crowd_category'].value_counts().to_dict()
        # Ensure at least 3 High, 2 Medium, 2 Low
        needed = {'High': 3, 'Medium': 2, 'Low': 2}
        for cat, min_count in needed.items():
            current = counts.get(cat, 0)
            if current < min_count:
                # Patch by duplicating and adjusting existing rows
                to_patch = subset.sample(n=min_count-current, replace=True)
                if cat == 'High':
                    to_patch['crowdlevel'] = np.random.uniform(75, 95, size=len(to_patch))
                elif cat == 'Medium':
                    to_patch['crowdlevel'] = np.random.uniform(40, 60, size=len(to_patch))
                else:
                    to_patch['crowdlevel'] = np.random.uniform(10, 25, size=len(to_patch))
                to_patch['crowd_category'] = cat
                df = pd.concat([df, to_patch], ignore_index=True)
                change_count += len(to_patch)

# Drop helper column and save
print(f"Patched {change_count} rows to ensure 3 High, 2 Medium, 2 Low per district/time_slot.")
df = df.drop(columns=['crowd_category'])
df.to_csv(csv_path, index=False) 