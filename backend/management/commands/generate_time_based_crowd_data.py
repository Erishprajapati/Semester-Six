from django.core.management.base import BaseCommand
from backend.models import Place
import csv
from datetime import datetime, timedelta
import random
import numpy as np

class Command(BaseCommand):
    help = 'Generate enhanced time-based crowd data with place-specific overrides, festival boosts, and district scaling'

    def handle(self, *args, **options):
        filename = 'enhanced_time_based_crowd_data.csv'
        self.stdout.write(f'Generating enhanced time-based crowd data to {filename}...')
        
        places = Place.objects.all()
        if not places.exists():
            self.stdout.write(self.style.ERROR('No places found in the database!'))
            return

        # Define logical crowd patterns based on place types and time slots
        CROWD_PATTERNS = {
            # Religious Places (Temples, Monasteries, etc.)
            'Religious': {
                'morning': {
                    'base': 75,  # High morning attendance for prayers
                    'weekend_boost': 15,  # Even higher on weekends
                    'holiday_boost': 20,  # Special festivals
                    'weather_impact': {'Sunny': 5, 'Cloudy': 0, 'Rainy': -10, 'Foggy': -5}
                },
                'afternoon': {
                    'base': 45,  # Moderate afternoon attendance
                    'weekend_boost': 10,
                    'holiday_boost': 15,
                    'weather_impact': {'Sunny': 0, 'Cloudy': 0, 'Rainy': -5, 'Foggy': 0}
                },
                'evening': {
                    'base': 60,  # Evening prayers and aarti
                    'weekend_boost': 10,
                    'holiday_boost': 15,
                    'weather_impact': {'Sunny': 0, 'Cloudy': 0, 'Rainy': -5, 'Foggy': 0}
                }
            },
            
            # Parks and Nature Areas
            'Nature': {
                'morning': {
                    'base': 65,  # Morning walkers, joggers
                    'weekend_boost': 20,  # Family outings
                    'holiday_boost': 15,
                    'weather_impact': {'Sunny': 10, 'Cloudy': 5, 'Rainy': -20, 'Foggy': -10}
                },
                'afternoon': {
                    'base': 35,  # Hot weather reduces attendance
                    'weekend_boost': 15,
                    'holiday_boost': 10,
                    'weather_impact': {'Sunny': -5, 'Cloudy': 5, 'Rainy': -15, 'Foggy': 0}
                },
                'evening': {
                    'base': 70,  # Evening walkers, families
                    'weekend_boost': 25,  # Peak evening activity
                    'holiday_boost': 20,
                    'weather_impact': {'Sunny': 5, 'Cloudy': 0, 'Rainy': -10, 'Foggy': -5}
                }
            },
            
            # Markets and Shopping Areas
            'Market': {
                'morning': {
                    'base': 25,  # Early morning shoppers
                    'weekend_boost': 15,
                    'holiday_boost': 20,
                    'weather_impact': {'Sunny': 0, 'Cloudy': 0, 'Rainy': -5, 'Foggy': 0}
                },
                'afternoon': {
                    'base': 75,  # Peak shopping hours
                    'weekend_boost': 20,  # Weekend shopping rush
                    'holiday_boost': 25,
                    'weather_impact': {'Sunny': 0, 'Cloudy': 0, 'Rainy': -10, 'Foggy': 0}
                },
                'evening': {
                    'base': 85,  # Evening shopping peak
                    'weekend_boost': 15,
                    'holiday_boost': 20,
                    'weather_impact': {'Sunny': 0, 'Cloudy': 0, 'Rainy': -15, 'Foggy': 0}
                }
            },
            
            # Tourist Attractions and Heritage Sites
            'Tourist': {
                'morning': {
                    'base': 40,  # Early tourists
                    'weekend_boost': 10,
                    'holiday_boost': 15,
                    'weather_impact': {'Sunny': 10, 'Cloudy': 5, 'Rainy': -15, 'Foggy': -5}
                },
                'afternoon': {
                    'base': 80,  # Peak tourist hours
                    'weekend_boost': 15,
                    'holiday_boost': 20,
                    'weather_impact': {'Sunny': 5, 'Cloudy': 0, 'Rainy': -20, 'Foggy': -10}
                },
                'evening': {
                    'base': 60,  # Evening tourists, sunset views
                    'weekend_boost': 10,
                    'holiday_boost': 15,
                    'weather_impact': {'Sunny': 10, 'Cloudy': 5, 'Rainy': -15, 'Foggy': -5}
                }
            },
            
            # Cultural and Museum Sites
            'Cultural': {
                'morning': {
                    'base': 30,  # Early visitors
                    'weekend_boost': 15,
                    'holiday_boost': 20,
                    'weather_impact': {'Sunny': 0, 'Cloudy': 0, 'Rainy': 5, 'Foggy': 0}  # Indoor activity
                },
                'afternoon': {
                    'base': 70,  # Peak cultural visit hours
                    'weekend_boost': 20,
                    'holiday_boost': 25,
                    'weather_impact': {'Sunny': 0, 'Cloudy': 0, 'Rainy': 10, 'Foggy': 0}  # Rain drives people indoors
                },
                'evening': {
                    'base': 50,  # Evening cultural activities
                    'weekend_boost': 15,
                    'holiday_boost': 20,
                    'weather_impact': {'Sunny': 0, 'Cloudy': 0, 'Rainy': 5, 'Foggy': 0}
                }
            },
            
            # Entertainment and Nightlife Areas (like Thamel)
            'Entertainment': {
                'morning': {
                    'base': 15,  # Very low morning activity
                    'weekend_boost': 5,
                    'holiday_boost': 10,
                    'weather_impact': {'Sunny': 0, 'Cloudy': 0, 'Rainy': 0, 'Foggy': 0}
                },
                'afternoon': {
                    'base': 45,  # Moderate afternoon activity
                    'weekend_boost': 15,
                    'holiday_boost': 20,
                    'weather_impact': {'Sunny': 5, 'Cloudy': 0, 'Rainy': -5, 'Foggy': 0}
                },
                'evening': {
                    'base': 90,  # Peak nightlife activity
                    'weekend_boost': 10,  # Already high, so smaller boost
                    'holiday_boost': 15,
                    'weather_impact': {'Sunny': 0, 'Cloudy': 0, 'Rainy': -10, 'Foggy': 0}
                }
            },
            
            # Historical and Heritage Sites
            'Historical': {
                'morning': {
                    'base': 35,  # Early heritage visitors
                    'weekend_boost': 10,
                    'holiday_boost': 15,
                    'weather_impact': {'Sunny': 10, 'Cloudy': 5, 'Rainy': -10, 'Foggy': -5}
                },
                'afternoon': {
                    'base': 75,  # Peak heritage visit hours
                    'weekend_boost': 15,
                    'holiday_boost': 20,
                    'weather_impact': {'Sunny': 5, 'Cloudy': 0, 'Rainy': -15, 'Foggy': -5}
                },
                'evening': {
                    'base': 55,  # Evening heritage visits
                    'weekend_boost': 10,
                    'holiday_boost': 15,
                    'weather_impact': {'Sunny': 5, 'Cloudy': 0, 'Rainy': -10, 'Foggy': -5}
                }
            }
        }

        # Default pattern for unknown categories
        DEFAULT_PATTERN = {
            'morning': {'base': 40, 'weekend_boost': 10, 'holiday_boost': 15, 'weather_impact': {'Sunny': 5, 'Cloudy': 0, 'Rainy': -10, 'Foggy': -5}},
            'afternoon': {'base': 60, 'weekend_boost': 15, 'holiday_boost': 20, 'weather_impact': {'Sunny': 0, 'Cloudy': 0, 'Rainy': -10, 'Foggy': 0}},
            'evening': {'base': 65, 'weekend_boost': 15, 'holiday_boost': 20, 'weather_impact': {'Sunny': 5, 'Cloudy': 0, 'Rainy': -10, 'Foggy': -5}}
        }

        # 1. PLACE_OVERRIDES: Place-specific crowd boosts for particular time slots and seasons
        PLACE_OVERRIDES = {
            # Major temples with special significance
            'Pashupatinath Temple': {
                'morning': {'Spring': 15, 'Autumn': 20, 'Winter': 10},  # Peak prayer times
                'evening': {'Spring': 10, 'Autumn': 15, 'Winter': 5},   # Evening aarti
                'special_boost': 25  # Always gets extra boost due to significance
            },
            'Swayambhunath Stupa': {
                'morning': {'Spring': 20, 'Autumn': 25, 'Winter': 15},  # Sunrise prayers
                'evening': {'Spring': 15, 'Autumn': 20, 'Winter': 10},  # Sunset prayers
                'special_boost': 20
            },
            'Boudhanath Stupa': {
                'morning': {'Spring': 15, 'Autumn': 20, 'Winter': 10},
                'evening': {'Spring': 10, 'Autumn': 15, 'Winter': 5},
                'special_boost': 18
            },
            
            # Major markets with unique patterns
            'Thamel': {
                'morning': {'Summer': -10, 'Winter': 5},  # Summer heat reduces morning activity
                'afternoon': {'Summer': -15, 'Winter': 10},  # Summer heat significantly reduces afternoon
                'evening': {'Summer': 5, 'Winter': 15},  # Evening is always busy
                'special_boost': 30  # Major tourist hub
            },
            'Asan Bazaar': {
                'morning': {'Spring': 20, 'Autumn': 25},  # Peak local shopping
                'afternoon': {'Summer': -10, 'Winter': 5},
                'evening': {'Spring': 15, 'Autumn': 20},
                'special_boost': 25
            },
            
            # Tourist hotspots
            'Kathmandu Durbar Square': {
                'morning': {'Spring': 10, 'Autumn': 15, 'Winter': 5},
                'afternoon': {'Spring': 15, 'Autumn': 20, 'Winter': 10},
                'evening': {'Spring': 10, 'Autumn': 15, 'Winter': 5},
                'special_boost': 20
            },
            'Patan Durbar Square': {
                'morning': {'Spring': 8, 'Autumn': 12, 'Winter': 3},
                'afternoon': {'Spring': 12, 'Autumn': 18, 'Winter': 8},
                'evening': {'Spring': 8, 'Autumn': 12, 'Winter': 3},
                'special_boost': 15
            },
            
            # Nature and parks
            'Garden of Dreams': {
                'morning': {'Spring': 25, 'Autumn': 30, 'Summer': 15, 'Winter': 10},
                'afternoon': {'Spring': 10, 'Autumn': 15, 'Summer': -10, 'Winter': 5},
                'evening': {'Spring': 20, 'Autumn': 25, 'Summer': 10, 'Winter': 15},
                'special_boost': 15
            },
            'Shivapuri Nagarjun National Park': {
                'morning': {'Spring': 30, 'Autumn': 35, 'Summer': 20, 'Winter': 15},
                'afternoon': {'Spring': 15, 'Autumn': 20, 'Summer': -5, 'Winter': 10},
                'evening': {'Spring': 25, 'Autumn': 30, 'Summer': 15, 'Winter': 20},
                'special_boost': 20
            }
        }

        # 2. FESTIVAL_PLACE_BOOSTS: Crowd boosts on specific festival dates for relevant places
        FESTIVAL_PLACE_BOOSTS = {
            # Dashain (Major Hindu festival - October)
            '2025-10-20': {
                'Pashupatinath Temple': 50,      # Major temple gets huge boost
                'Swayambhunath Stupa': 40,       # Buddhist site also gets boost
                'Boudhanath Stupa': 35,          # Buddhist site
                'Kathmandu Durbar Square': 30,   # Cultural celebrations
                'Patan Durbar Square': 25,       # Cultural celebrations
                'Thamel': 20,                    # Tourist area
                'Asan Bazaar': 35,               # Shopping for festival
                'Garden of Dreams': 15,          # Family outings
                'default_boost': 10              # All other places get small boost
            },
            
            # Tihar (Festival of Lights - November)
            '2025-11-09': {
                'Pashupatinath Temple': 45,
                'Swayambhunath Stupa': 35,
                'Boudhanath Stupa': 30,
                'Kathmandu Durbar Square': 25,
                'Patan Durbar Square': 20,
                'Thamel': 25,                    # Lights and celebrations
                'Asan Bazaar': 30,               # Shopping for festival
                'Garden of Dreams': 20,          # Evening light displays
                'default_boost': 8
            },
            
            # Nepali New Year (April)
            '2025-04-14': {
                'Pashupatinath Temple': 30,
                'Swayambhunath Stupa': 25,
                'Boudhanath Stupa': 20,
                'Kathmandu Durbar Square': 35,   # Major celebrations
                'Patan Durbar Square': 30,       # Cultural celebrations
                'Thamel': 40,                    # Tourist celebrations
                'Asan Bazaar': 25,               # Shopping
                'Garden of Dreams': 30,          # Family celebrations
                'default_boost': 15
            },
            
            # Janai Purnima (August)
            '2025-08-19': {
                'Pashupatinath Temple': 40,      # Major ritual site
                'Swayambhunath Stupa': 30,
                'Boudhanath Stupa': 25,
                'Kathmandu Durbar Square': 20,
                'Patan Durbar Square': 15,
                'Thamel': 15,
                'Asan Bazaar': 20,
                'default_boost': 5
            }
        }

        # 3. DISTRICT_SCALING: Scale crowd levels based on district's population intensity
        DISTRICT_SCALING = {
            'Kathmandu': 1.2,    # Most populous and busy district
            'Lalitpur': 1.1,     # Second most populous
            'Bhaktapur': 1.0,    # Standard scaling
            'default': 1.0       # Default scaling for unknown districts
        }

        # Weather patterns by season
        weather_by_season = {
            'Winter': (['Sunny', 'Cloudy', 'Foggy'], [0.3, 0.4, 0.3]),
            'Spring': (['Sunny', 'Cloudy', 'Rainy'], [0.4, 0.3, 0.3]),
            'Summer': (['Sunny', 'Cloudy', 'Rainy'], [0.5, 0.3, 0.2]),
            'Autumn': (['Sunny', 'Cloudy', 'Rainy', 'Foggy'], [0.3, 0.3, 0.2, 0.2])
        }

        # Major holidays and festivals (expanded list)
        holidays = set([
            '2025-01-01',  # New Year
            '2025-04-14',  # Nepali New Year
            '2025-05-01',  # Labor Day
            '2025-08-19',  # Janai Purnima
            '2025-09-19',  # Constitution Day
            '2025-10-20',  # Dashain
            '2025-11-09',  # Tihar
            '2025-12-25',  # Christmas
        ])

        # Generate data for 6 months
        end_date = datetime.now()
        start_date = end_date - timedelta(days=180)
        
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([
                'place_id', 'category', 'district', 'time_slot', 'hour', 'day_of_week', 
                'month', 'season', 'is_weekend', 'is_holiday', 'weather_condition', 'crowdlevel'
            ])
            
            total_records = 0
            
            for place in places:
                self.stdout.write(f'Processing {place.name} ({place.category})...')
                
                # Get pattern for this place category
                category_pattern = CROWD_PATTERNS.get(place.category, DEFAULT_PATTERN)
                
                # Get place-specific overrides
                place_overrides = PLACE_OVERRIDES.get(place.name, {})
                
                # Get district scaling factor
                district_scaling = DISTRICT_SCALING.get(place.district, DISTRICT_SCALING['default'])
                
                current_date = start_date
                while current_date <= end_date:
                    # Determine season
                    month = current_date.month
                    if month in [12, 1, 2]:
                        season = 'Winter'
                    elif month in [3, 4, 5]:
                        season = 'Spring'
                    elif month in [6, 7, 8]:
                        season = 'Summer'
                    else:
                        season = 'Autumn'
                    
                    # Check if it's a holiday
                    is_holiday = current_date.strftime('%Y-%m-%d') in holidays
                    
                    # Check if it's weekend
                    is_weekend = current_date.weekday() >= 5
                    
                    # Get weather for this day
                    weather_options, weather_probs = weather_by_season[season]
                    weather = np.random.choice(weather_options, p=weather_probs)
                    
                    # Generate data for each time slot
                    time_slots = [
                        ('morning', 8),    # 8 AM for morning
                        ('afternoon', 14), # 2 PM for afternoon
                        ('evening', 19)    # 7 PM for evening
                    ]
                    
                    for time_slot, hour in time_slots:
                        # Get base pattern for this time slot
                        slot_pattern = category_pattern[time_slot]
                        
                        # 5. Enhanced crowd level calculation with all new factors
                        crowd_level = slot_pattern['base']
                        
                        # Apply weekend boost
                        if is_weekend:
                            crowd_level += slot_pattern['weekend_boost']
                        
                        # Apply holiday boost
                        if is_holiday:
                            crowd_level += slot_pattern['holiday_boost']
                        
                        # Apply weather impact
                        weather_impact = slot_pattern['weather_impact'].get(weather, 0)
                        crowd_level += weather_impact
                        
                        # Apply place-specific overrides
                        if time_slot in place_overrides:
                            time_slot_overrides = place_overrides[time_slot]
                            if season in time_slot_overrides:
                                crowd_level += time_slot_overrides[season]
                        
                        # Apply special boost for significant places
                        if 'special_boost' in place_overrides:
                            crowd_level += place_overrides['special_boost']
                        
                        # Apply festival-specific boosts
                        current_date_str = current_date.strftime('%Y-%m-%d')
                        if current_date_str in FESTIVAL_PLACE_BOOSTS:
                            festival_boosts = FESTIVAL_PLACE_BOOSTS[current_date_str]
                            if place.name in festival_boosts:
                                crowd_level += festival_boosts[place.name]
                            elif 'default_boost' in festival_boosts:
                                crowd_level += festival_boosts['default_boost']
                        
                        # Apply district scaling
                        crowd_level *= district_scaling
                        
                        # 6. Enhanced randomness: ±15% on weekends, ±10% on weekdays
                        if is_weekend:
                            random_factor = random.uniform(-0.15, 0.15)  # ±15% on weekends
                        else:
                            random_factor = random.uniform(-0.10, 0.10)  # ±10% on weekdays
                        crowd_level *= (1 + random_factor)
                        
                        # Ensure crowd level is within bounds (0-100%)
                        crowd_level = max(0, min(100, crowd_level))
                        
                        # Write record
                        writer.writerow([
                            place.id,
                            place.category,
                            place.district,
                            time_slot,
                            hour,
                            current_date.weekday(),
                            month,
                            season,
                            1 if is_weekend else 0,
                            1 if is_holiday else 0,
                            weather,
                            round(crowd_level, 1)
                        ])
                        
                        total_records += 1
                    
                    current_date += timedelta(days=1)
        
        self.stdout.write(
            self.style.SUCCESS(
                f'Successfully generated {total_records:,} enhanced time-based crowd records!\n'
                f'File saved as: {filename}\n\n'
                f'🆕 NEW ENHANCEMENTS APPLIED:\n'
                f'• Place-specific overrides for {len(PLACE_OVERRIDES)} major places\n'
                f'• Festival boosts for {len(FESTIVAL_PLACE_BOOSTS)} major festivals\n'
                f'• District scaling factors: {DISTRICT_SCALING}\n'
                f'• Enhanced randomness: ±15% on weekends, ±10% on weekdays\n'
                f'• All factors combined logically with 0-100% bounds\n\n'
                f'Time Slot Definitions:\n'
                f'• Morning: 5:00 AM - 10:00 AM (peak activity for temples, parks, morning routines)\n'
                f'• Afternoon: 10:00 AM - 5:00 PM (peak for markets, tourist attractions, cultural sites)\n'
                f'• Evening: 5:00 PM - 10:00 PM (peak for nightlife, evening walks, entertainment)\n\n'
                f'Logical Patterns Applied:\n'
                f'• Religious places: High morning attendance for prayers, moderate afternoon/evening\n'
                f'• Nature/Parks: Morning walkers, evening families, low afternoon due to heat\n'
                f'• Markets: Low morning, peak afternoon/evening shopping hours\n'
                f'• Tourist sites: Moderate morning, peak afternoon, good evening for sunset views\n'
                f'• Cultural sites: Indoor activity, rain actually increases attendance\n'
                f'• Entertainment (Thamel): Very low morning, moderate afternoon, peak evening/nightlife\n'
                f'• Historical sites: Similar to tourist patterns but more focused on heritage\n\n'
                f'Next steps:\n'
                f'1. Run: python manage.py train_improved_crowd_model --csv-file {filename}\n'
                f'2. Test predictions with different time slots and weather conditions\n'
            )
        ) 