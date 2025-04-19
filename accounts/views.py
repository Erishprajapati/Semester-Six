from django.shortcuts import render,redirect
from django.contrib.auth.models import User
from django.contrib import messages
from django.contrib.auth import authenticate, login,logout
from django.contrib.auth.decorators import login_required
import re
from backend.models import *

# Create your views here.
def register_user(request):
    if request.method == 'POST':
        username = request.POST['username']
        email = request.POST['email']
        password = request.POST['password']
        confirm_password = request.POST['confirm_password']

        # Check if passwords match
        if password != confirm_password:
            messages.error(request, "Passwords do not match.")
            return redirect('register')

        # Check password strength (min 8 chars, 1 uppercase, 1 number)
        if not re.match(r'^(?=.*[A-Z])(?=.*\d).{8,}$', password):
            messages.error(request, "Password must be at least 8 characters long, contain one uppercase letter and one number.")
            return redirect('register')

        # Validate email format
        if not re.match(r'^[\w\.-]+@[\w\.-]+\.\w+$', email):
            messages.error(request, "Please enter a valid email address.")
            return redirect('register')

        # Check if username already exists
        if User.objects.filter(username=username).exists():
            messages.error(request, "Username already taken.")
            return redirect('register')

        # Check if email already exists
        if User.objects.filter(email=email).exists():
            messages.error(request, "Email already registered.")
            return redirect('register')

        # All good, create user
        user = User.objects.create_user(username=username, email=email, password=password)
        user.save()
        messages.success(request, "Registration successful! You can now log in.")
        return redirect('login')

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


@login_required(login_url='login')
def home(request):
    query = request.GET.get('q', '')  # search query from input
    places = Place.objects.all()

    if query:
        places = places.filter(name__icontains=query)

    place_name = "Kathmandu"  # You can change this or make it dynamic
    try:
        place = Place.objects.get(name__iexact=place_name)
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
        }
    except Place.DoesNotExist:
        context = {
            'place_name': '',
            'timestamps': [],
            'crowd_levels': [],
            'places': [],
            'query': query,
            'error': f"{place_name} not found."
        }

    return render(request, 'map.html', context)
