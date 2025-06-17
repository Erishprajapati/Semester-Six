from django.shortcuts import render,redirect
from django.contrib.auth.models import User
from django.contrib import messages
from django.contrib.auth import authenticate, login,logout
from django.contrib.auth.decorators import login_required
import re
from backend.models import *
from django.utils import timezone
import joblib
import os
from datetime import datetime

# Create your views here.
def register_user(request):
    if request.method == 'POST':
        username = request.POST['username']
        email = request.POST['email']
        password = request.POST['password']
        confirm_password = request.POST['confirm_password']

        # Check if passwords match
        if password != confirm_password:
            messages.error(request, "Passwords do not match. Please try again.")
            return redirect('register')

        # Check password strength (min 8 chars, 1 uppercase, 1 number)
        if not re.match(r'^(?=.*[A-Z])(?=.*\d).{8,}$', password):
            messages.error(request, "Password must be at least 8 characters long and contain one uppercase letter and one number.")
            return redirect('register')

        # Validate email format
        if not re.match(r'^[\w\.-]+@[\w\.-]+\.\w+$', email):
            messages.error(request, "Please enter a valid email address.")
            return redirect('register')

        # Check if username already exists
        if User.objects.filter(username=username).exists():
            messages.error(request, "Username is already taken. Please choose a different username.")
            return redirect('register')

        # Check if email already exists
        if User.objects.filter(email=email).exists():
            messages.error(request, "This email is already registered. Please use a different email or try logging in.")
            return redirect('register')

        # All good, create user
        try:
            user = User.objects.create_user(username=username, email=email, password=password)
            user.save()
            messages.success(request, "Registration successful! You can now log in with your credentials.")
            return redirect('login')
        except Exception as e:
            messages.error(request, "An error occurred during registration. Please try again.")
            return redirect('register')

    return render(request, 'register.html')
        
# def login_user(request):
#     if request.method == 'POST':
#         username = request.POST['username']
#         password = request.POST['password']

#         user = authenticate(request, username=username, password=password)

#         if user is not None:
#             login(request, user)
#             return redirect('dashboard')
#         else:
#             messages.error(request, 'Invalid username or password')
#             return redirect('login')

#     return render(request, 'login.html')

def login_user(request):
    if request.method == 'POST':
        username = request.POST['username']
        password = request.POST['password']

        # DEBUG: print info
        print("Username:", username)
        print("Password:", password)

        try:
            user_check = User.objects.get(username=username)
            print("User found in DB:", user_check)
        except User.DoesNotExist:
            print("User does not exist!")

        user = authenticate(request, username=username, password=password)

        if user is not None:
            login(request, user)
            return redirect('dashboard')  # <- make sure this matches your URL name
        else:
            messages.error(request, 'Invalid username or password')
            return redirect('login')

    return render(request, 'login.html')
# Logout
def logout_user(request):
    logout(request)
    messages.success(request, 'You have been logged out.')
    return redirect('login')

# Helper function to get season
def get_season(month):
    if month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    elif month in [9, 10, 11]:
        return 'Fall'
    else:
        return 'Winter'

# Helper function to get time slot based on current time
def get_current_time_slot():
    current_hour = datetime.now().hour
    if 6 <= current_hour < 12:
        return 'morning'
    elif 12 <= current_hour < 18:
        return 'afternoon'
    else:
        return 'evening'

@login_required(login_url='login')
def home(request):
    query = request.GET.get('q', '')  # search query from input
    places = Place.objects.all()

    if query:
        places = places.filter(name__icontains=query)

    place_name = "Kathmandu"  # You can change this or make it dynamic
    try:
        place = Place.objects.get(name__iexact=place_name)
        today = timezone.now().date()
        current_time_slot = get_current_time_slot()
        
        # Load the trained model for predictions
        model_path = 'crowd_prediction_model.pkl'
        if os.path.exists(model_path):
            try:
                model_data = joblib.load(model_path)
                model = model_data['model']
                encoders = model_data['label_encoders']
                
                # Get current time features
                current_time = datetime.now()
                day_of_week = current_time.weekday()
                month = current_time.month
                season = get_season(month)
                is_weekend = 1 if day_of_week >= 5 else 0
                is_holiday = 0  # You can add holiday logic here
                weather_condition = 'Sunny'  # Default weather
                
                # Predict crowd levels for all time slots
                time_slots = ['morning', 'afternoon', 'evening']
                predicted_data = {}
                
                for time_slot in time_slots:
                    # Encode categorical features
                    features = [
                        place.id,
                        encoders['category'].transform([place.category])[0],
                        encoders['district'].transform([place.district])[0],
                        encoders['time_slot'].transform([time_slot])[0],
                        day_of_week,
                        month,
                        encoders['season'].transform([season])[0],
                        is_weekend,
                        is_holiday,
                        encoders['weather_condition'].transform([weather_condition])[0]
                    ]
                    
                    # Make prediction
                    predicted_crowd = model.predict([features])[0]
                    predicted_crowd = float(max(0, min(100, round(predicted_crowd, 1))))
                    
                    # Determine status
                    if predicted_crowd > 70:
                        status = 'High'
                    elif predicted_crowd > 30:
                        status = 'Medium'
                    else:
                        status = 'Low'
                    
                    predicted_data[time_slot] = {
                        'crowdlevel': predicted_crowd,
                        'status': status
                    }
                
                # Create mock CrowdData objects for template compatibility
                class MockCrowdData:
                    def __init__(self, crowdlevel, status):
                        self.crowdlevel = crowdlevel
                        self.status = status
                
                morning_data = MockCrowdData(predicted_data['morning']['crowdlevel'], predicted_data['morning']['status'])
                afternoon_data = MockCrowdData(predicted_data['afternoon']['crowdlevel'], predicted_data['afternoon']['status'])
                evening_data = MockCrowdData(predicted_data['evening']['crowdlevel'], predicted_data['evening']['status'])
                
                # Create time-based chart data (last 10 hours with realistic variations)
                timestamps = []
                crowd_levels = []
                current_hour = current_time.hour
                
                for i in range(10):
                    hour = (current_hour - 9 + i) % 24
                    time_slot = get_current_time_slot() if hour == current_hour else ('morning' if 6 <= hour < 12 else 'afternoon' if 12 <= hour < 18 else 'evening')
                    
                    # Get predicted crowd for this time slot
                    if time_slot in predicted_data:
                        crowd_level = predicted_data[time_slot]['crowdlevel']
                        # Add some variation for realism
                        import random
                        variation = random.uniform(-5, 5)
                        crowd_level = max(0, min(100, crowd_level + variation))
                    else:
                        crowd_level = 50  # Default
                    
                    timestamps.append(f"{hour:02d}:00")
                    crowd_levels.append(round(crowd_level, 1))
                
            except Exception as e:
                print(f"Model prediction error: {e}")
                # Fallback to static data if model fails
                morning_data = CrowdData.objects.filter(place=place, time_slot='morning').order_by('-timestamp').first()
                afternoon_data = CrowdData.objects.filter(place=place, time_slot='afternoon').order_by('-timestamp').first()
                evening_data = CrowdData.objects.filter(place=place, time_slot='evening').order_by('-timestamp').first()
                
                # Get historical data for the chart
                last_10 = CrowdData.objects.filter(place=place).order_by('-timestamp')[:10]
                last_10 = list(reversed(last_10))
                timestamps = [cd.timestamp.strftime('%H:%M') for cd in last_10]
                crowd_levels = [cd.crowdlevel for cd in last_10]
        else:
            # Fallback to static data if model file doesn't exist
            morning_data = CrowdData.objects.filter(place=place, time_slot='morning').order_by('-timestamp').first()
            afternoon_data = CrowdData.objects.filter(place=place, time_slot='afternoon').order_by('-timestamp').first()
            evening_data = CrowdData.objects.filter(place=place, time_slot='evening').order_by('-timestamp').first()
            
            # Get historical data for the chart
            last_10 = CrowdData.objects.filter(place=place).order_by('-timestamp')[:10]
            last_10 = list(reversed(last_10))
            timestamps = [cd.timestamp.strftime('%H:%M') for cd in last_10]
            crowd_levels = [cd.crowdlevel for cd in last_10]

        context = {
            'place_name': place.name,
            'timestamps': timestamps,
            'crowd_levels': crowd_levels,
            'places': places,
            'query': query,
            'morning_data': morning_data,
            'afternoon_data': afternoon_data,
            'evening_data': evening_data,
            'today': today.strftime('%Y-%m-%d'),
            'current_time_slot': current_time_slot,
            'is_predicted_data': 'model' in locals(),
        }
    except Place.DoesNotExist:
        context = {
            'place_name': '',
            'timestamps': [],
            'crowd_levels': [],
            'places': [],
            'query': query,
            'error': f"{place_name} not found.",
            'morning_data': None,
            'afternoon_data': None,
            'evening_data': None,
            'today': timezone.now().date().strftime('%Y-%m-%d'),
            'current_time_slot': get_current_time_slot(),
            'is_predicted_data': False,
        }

    return render(request, 'map.html', context)

def graph(request):
    return render(request, 'graph.html')

@login_required
def update_profile(request):
    if request.method == 'POST':
        user = request.user
        username = request.POST['username']
        email = request.POST['email']
        new_password = request.POST.get('new_password')
        confirm_password = request.POST.get('confirm_password')

        user.username = username
        user.email = email

        if new_password:
            if new_password == confirm_password:
                user.set_password(new_password)
            else:
                messages.error(request, "Passwords do not match.")
                return redirect('update_profile')  # Or re-render the form with error

        user.save()
        messages.success(request, "Profile updated.")
        return redirect('login')  # Redirect to login page

    return render(request, 'accounts/profile.html')
