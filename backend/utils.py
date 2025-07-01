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

# New optimized weather functions
@cache_result(timeout=600)  # Cache for 10 minutes
def get_batch_weather_data(places):
    """
    Get weather data for multiple places in a single optimized request
    Uses a single API call to get weather for Kathmandu area (all places are in Kathmandu valley)
    """
    try:
        # Since all places are in Kathmandu valley, use Kathmandu coordinates for weather
        # This reduces API calls from N (number of places) to 1
        kathmandu_lat = 27.7172
        kathmandu_lon = 85.3240
        
        weather_data = get_weather(kathmandu_lat, kathmandu_lon)
        
        if weather_data and 'weather' in weather_data and 'main' in weather_data:
            weather_main = weather_data['weather'][0]['main'].lower()
            
            # Map OpenWeatherMap conditions to our model's categories
            if weather_main in ['clear']:
                weather_condition = 'Sunny'
            elif weather_main in ['clouds', 'cloudy']:
                weather_condition = 'Cloudy'
            elif weather_main in ['rain', 'drizzle', 'snow', 'thunderstorm']:
                weather_condition = 'Rainy'
            elif weather_main in ['fog', 'mist', 'haze']:
                weather_condition = 'Foggy'
            else:
                weather_condition = 'Sunny'
        else:
            weather_condition = 'Sunny'
        
        # Return the same weather condition for all places
        return {place.id: weather_condition for place in places}
        
    except Exception as e:
        print(f"Error getting batch weather data: {e}")
        # Return default weather for all places
        return {place.id: 'Sunny' for place in places}

def get_optimized_weather_for_places(places):
    """
    Optimized function to get weather for multiple places
    Uses batching and caching to reduce API calls
    """
    if not places:
        return {}
    
    # Use batch weather data (single API call for Kathmandu area)
    return get_batch_weather_data(places)

# Performance monitoring
def log_performance(func_name, start_time, end_time, details=""):
    """Log performance metrics for debugging"""
    duration = end_time - start_time
    print(f"[PERFORMANCE] {func_name}: {duration:.2f}s {details}")
