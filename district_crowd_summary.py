import pandas as pd

# Load your data
try:
    df = pd.read_csv("enhanced_crowd_data.csv")
except Exception as e:
    print(f"Error loading CSV: {e}")
    exit(1)

# User input for district and time slot
searched_district = input("Enter district name: ").strip()
searched_time = input("Enter time slot (morning/afternoon/evening): ").strip().lower()

# Filter by district and time slot
district_df = df[
    (df["district"].str.lower() == searched_district.lower()) &
    (df["time_slot"].str.lower() == searched_time)
]

if district_df.empty:
    print(f"⚠️ No data found for {searched_district} ({searched_time}). Try another district or time slot.")
    exit(0)

# Group by place to get average crowd per place
grouped = (
    district_df.groupby(["place_id", "category"])["crowdlevel"]
    .mean()
    .reset_index()
)

# Classify into high / medium / low
def classify(level):
    if level >= 75:
        return "high"
    elif level >= 40:
        return "medium"
    else:
        return "low"

grouped["crowd_category"] = grouped["crowdlevel"].apply(classify)

# Select top 3-2-2 if enough data exists
high = grouped[grouped["crowd_category"] == "high"].sort_values("crowdlevel", ascending=False).head(3)
medium = grouped[grouped["crowd_category"] == "medium"].sort_values("crowdlevel", ascending=False).head(2)
low = grouped[grouped["crowd_category"] == "low"].sort_values("crowdlevel", ascending=False).head(2)

# Combine all
final = pd.concat([high, medium, low])

# Show results
if not final.empty:
    print(f"\n📍 3-2-2 Crowd Summary for {searched_district.title()} ({searched_time.title()}):")
    for _, row in final.iterrows():
        print(f"{row['crowd_category'].capitalize()} ➜ Place ID: {row['place_id']} ({row['category']}) - {row['crowdlevel']:.1f}%")
else:
    print(f"⚠️ No data found for {searched_district} ({searched_time}). Try another district or time slot.") 