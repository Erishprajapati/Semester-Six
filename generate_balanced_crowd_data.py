import csv
import random

# Curated list of famous places for each district
CURATED_PLACES = [
    # Kathmandu
    {'place_name': 'Kathmandu Durbar Square', 'category': 'Durbar Square', 'district': 'Kathmandu'},
    {'place_name': 'Hanuman Dhoka Palace', 'category': 'Palace', 'district': 'Kathmandu'},
    {'place_name': 'Kumari Ghar', 'category': 'Temple', 'district': 'Kathmandu'},
    {'place_name': 'Taleju Temple', 'category': 'Temple', 'district': 'Kathmandu'},
    {'place_name': 'Kasthamandap', 'category': 'Temple', 'district': 'Kathmandu'},
    {'place_name': 'Swayambhunath', 'category': 'Stupa', 'district': 'Kathmandu'},
    {'place_name': 'Pashupatinath Temple', 'category': 'Temple', 'district': 'Kathmandu'},
    {'place_name': 'Boudhanath Stupa', 'category': 'Stupa', 'district': 'Kathmandu'},
    {'place_name': 'Asan Market', 'category': 'Market', 'district': 'Kathmandu'},
    {'place_name': 'Freak Street', 'category': 'Market', 'district': 'Kathmandu'},
    {'place_name': 'Gaddi Baithak', 'category': 'Palace', 'district': 'Kathmandu'},
    {'place_name': 'Shiva Parvati Temple', 'category': 'Temple', 'district': 'Kathmandu'},
    {'place_name': 'Kaal Bhairav', 'category': 'Temple', 'district': 'Kathmandu'},
    {'place_name': 'Sweta Bhairav', 'category': 'Temple', 'district': 'Kathmandu'},
    {'place_name': 'Akash Bhairav Temple', 'category': 'Temple', 'district': 'Kathmandu'},
    {'place_name': 'Basantapur Tower', 'category': 'Tower', 'district': 'Kathmandu'},
    {'place_name': 'Jagannath Temple', 'category': 'Temple', 'district': 'Kathmandu'},
    {'place_name': 'Nasal Chowk', 'category': 'Palace', 'district': 'Kathmandu'},
    {'place_name': 'Hanuman Dhoka Museum', 'category': 'Museum', 'district': 'Kathmandu'},
    # Bhaktapur
    {'place_name': 'Bhaktapur Durbar Square', 'category': 'Durbar Square', 'district': 'Bhaktapur'},
    {'place_name': 'Nyatapola Temple', 'category': 'Temple', 'district': 'Bhaktapur'},
    {'place_name': 'Dattatreya Temple', 'category': 'Temple', 'district': 'Bhaktapur'},
    {'place_name': 'Pottery Square', 'category': 'Market', 'district': 'Bhaktapur'},
    {'place_name': 'Taumadhi Square', 'category': 'Square', 'district': 'Bhaktapur'},
    {'place_name': 'Siddha Pokhari', 'category': 'Lake', 'district': 'Bhaktapur'},
    {'place_name': '55 Window Palace', 'category': 'Palace', 'district': 'Bhaktapur'},
    {'place_name': 'Vatsala Temple', 'category': 'Temple', 'district': 'Bhaktapur'},
    {'place_name': 'Bhairavnath Temple', 'category': 'Temple', 'district': 'Bhaktapur'},
    {'place_name': 'Changu Narayan Temple', 'category': 'Temple', 'district': 'Bhaktapur'},
    # Lalitpur (Patan)
    {'place_name': 'Patan Durbar Square', 'category': 'Durbar Square', 'district': 'Lalitpur'},
    {'place_name': 'Krishna Mandir', 'category': 'Temple', 'district': 'Lalitpur'},
    {'place_name': 'Hiranya Varna Mahavihar', 'category': 'Monastery', 'district': 'Lalitpur'},
    {'place_name': 'Mahabouddha Temple', 'category': 'Temple', 'district': 'Lalitpur'},
    {'place_name': 'Kumbeshwar Temple', 'category': 'Temple', 'district': 'Lalitpur'},
    {'place_name': 'Rudra Varna Mahavihar', 'category': 'Monastery', 'district': 'Lalitpur'},
    {'place_name': 'Ashok Stupa', 'category': 'Stupa', 'district': 'Lalitpur'},
    {'place_name': 'Patan Museum', 'category': 'Museum', 'district': 'Lalitpur'},
    {'place_name': 'Rato Machhindranath Temple', 'category': 'Temple', 'district': 'Lalitpur'},
    {'place_name': 'Sundari Chowk', 'category': 'Palace', 'district': 'Lalitpur'},
]

TIME_SLOTS = ['morning', 'afternoon', 'evening']

# Crowd level ranges
CROWD_RANGES = {
    'High': (75, 99),
    'Medium': (40, 65),
    'Low': (8, 28)
}

# Helper values for new columns
HOUR_MAP = {'morning': 9, 'afternoon': 14, 'evening': 19}
DAY_OF_WEEK_MAP = {'morning': 4, 'afternoon': 5, 'evening': 6}  # Friday, Saturday, Sunday
MONTH = 6  # June
SEASON = 'Summer'
WEATHER_OPTIONS = ['Sunny', 'Cloudy', 'Rainy', 'Foggy']

rows = []
place_id_counter = 1
for place in CURATED_PLACES:
    for time_slot in TIME_SLOTS:
        for group in ['High', 'Medium', 'Low']:
            crowdlevel = round(random.uniform(*CROWD_RANGES[group]), 1)
            # Add small random noise for visual distinction
            noise = random.uniform(-3, 3)
            crowdlevel = max(0, min(100, round(crowdlevel + noise, 2)))
            hour = HOUR_MAP[time_slot]
            day_of_week = DAY_OF_WEEK_MAP[time_slot]
            month = MONTH
            season = SEASON
            is_weekend = 1 if day_of_week in [5, 6] else 0
            weather_condition = random.choice(WEATHER_OPTIONS)
            rows.append({
                'place_id': place_id_counter,
                'place_name': place['place_name'],
                'category': place['category'],
                'district': place['district'],
                'time_slot': time_slot,
                'hour': hour,
                'day_of_week': day_of_week,
                'month': month,
                'season': season,
                'is_weekend': is_weekend,
                'weather_condition': weather_condition,
                'crowdlevel': crowdlevel,
                'crowd_category': group
            })
            place_id_counter += 1

with open('balanced_crowd_data.csv', 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=[
        'place_id', 'place_name', 'category', 'district', 'time_slot',
        'hour', 'day_of_week', 'month', 'season', 'is_weekend', 'weather_condition',
        'crowdlevel', 'crowd_category'])
    writer.writeheader()
    writer.writerows(rows)

print('balanced_crowd_data.csv generated with all required columns for model training and all crowd categories for every place/time slot.') 