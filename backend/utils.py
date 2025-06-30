import os
import requests
from dotenv import load_dotenv
from functools import wraps
import time
from django.core.cache import cache

load_dotenv()
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")

def cache_result(timeout=300):  # 5 minutes default cache
    """
    Decorator to cache function results for improved performance
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Create a cache key based on function name and arguments
            cache_key = f"{func.__name__}:{hash(str(args) + str(sorted(kwargs.items())))}"
            
            # Try to get cached result
            cached_result = cache.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            # If not cached, execute function and cache result
            result = func(*args, **kwargs)
            cache.set(cache_key, result, timeout)
            return result
        return wrapper
    return decorator

def get_weather(lat, lon):
    url = f"http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={WEATHER_API_KEY}&units=metric"
    response = requests.get(url)
    return response.json()

@cache_result(timeout=600)  # Cache weather data for 10 minutes
def get_real_time_weather_for_place(place):
    """
    Get real-time weather for a specific place and map it to our model's weather categories
    Returns: weather_condition (str) that matches our model's expected values
    """
    try:
        # Get weather data from OpenWeatherMap API
        weather_data = get_weather(place.latitude, place.longitude)
        
        if weather_data and 'weather' in weather_data and 'main' in weather_data:
            weather_main = weather_data['weather'][0]['main'].lower()
            
            # Map OpenWeatherMap conditions to our model's categories
            if weather_main in ['clear']:
                return 'Sunny'
            elif weather_main in ['clouds', 'cloudy']:
                return 'Cloudy'
            elif weather_main in ['rain', 'drizzle', 'snow', 'thunderstorm']:
                return 'Rainy'
            elif weather_main in ['fog', 'mist', 'haze']:
                return 'Foggy'
            else:
                return 'Sunny'  # Default fallback
        else:
            return 'Sunny'  # Default if API fails
            
    except Exception as e:
        print(f"Error getting weather for {place.name}: {e}")
        return 'Sunny'  # Default fallback

def get_weather_impact_on_crowd(weather_condition):
    """
    Calculate weather impact on crowd levels
    Returns: impact percentage (positive or negative)
    """
    weather_impacts = {
        'Sunny': 5,      # Good weather increases crowds
        'Cloudy': 0,     # No change
        'Rainy': -10,    # Rain reduces crowds
        'Foggy': -5      # Fog reduces visibility and crowds
    }
    
    return weather_impacts.get(weather_condition, 0)
