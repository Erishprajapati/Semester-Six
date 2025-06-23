import random
import json
from django.utils.timezone import now
from django.shortcuts import render, get_object_or_404,redirect
from django.http import JsonResponse
from rest_framework.response import Response
from rest_framework.decorators import api_view
from .models import *
from .serializers import PlaceSerializer, CrowdDataSerializer, TagSerializer
from .utils import get_weather
from django.contrib.auth.decorators import login_required
from django.views.decorators.csrf import csrf_exempt
from django.db.models import Max
from django.contrib import messages
from django.db.models import Q, Subquery, OuterRef
from math import radians, cos, sin, asin, sqrt
import joblib
import os
from datetime import datetime
import pandas as pd  # Add this at the top if not present
from django.urls import reverse
from django.contrib.auth import authenticate, login
from .ml_model import get_current_season, get_current_weather

# Create your views here.
def map_view(request):
    return render(request, 'Map.html')
    """it will act as key pair value"""

@login_required
def profile_view(request):
    user = request.user
    # Handle profile update (username, email, password)
    if request.method == "POST":
        username = request.POST.get("username")
        email = request.POST.get("email")
        new_password = request.POST.get("new_password")
        confirm_password = request.POST.get("confirm_password")
        selected_tags = request.POST.getlist("tags")

        # If tags are selected, redirect to places_by_tags view with correct api/ prefix
        if selected_tags:
            return redirect(f'/api/places-by-tags/?tags={"&tags=".join(selected_tags)}')

        # Validate username
        if not username:
            messages.error(request, "Username cannot be empty.")
            return redirect('profile_view')
        # Check for unique username if changed
        if username != user.username and User.objects.filter(username=username).exists():
            messages.error(request, "This username is already taken.")
            return redirect('profile_view')

        # Update username and email
        user.username = username
        user.email = email

        # Password update
        if new_password:
            if new_password == confirm_password:
                user.set_password(new_password)
                messages.success(request, "Password updated.")
            else:
                messages.error(request, "Passwords do not match.")
                return redirect('profile_view')

        user.save()
        # Handle user preferences (tags)
        from .models import UserPreference, Tag, Place
        user_pref, created = UserPreference.objects.get_or_create(user=user)
        if selected_tags:
            user_pref.tags.set(selected_tags)
        else:
            user_pref.tags.clear()
        user_pref.save()
        messages.success(request, "Profile updated successfully.")
        return redirect('profile_view')

    # Get user's search history
    search_history = SearchHistory.objects.filter(user=user)[:10]  # Get last 10 searches
    # Get all tags
    from .models import UserPreference, Tag, Place
    all_tags = Tag.objects.filter(place__isnull=False).distinct()
    user_pref, created = UserPreference.objects.get_or_create(user=user)
    preferred_tags = user_pref.tags.all()
    # Get recommended places based on preferred tags
    recommended_places = Place.objects.filter(tags__in=preferred_tags).distinct()

    return render(request, "profile.html", {
        'search_history': search_history,
        'all_tags': all_tags,
        'preferred_tags': preferred_tags,
        'recommended_places': recommended_places,
    })

# def places_by_category(request, category):
#     category_cleaned = category.replace('#', '').lower()

#     # Subquery: Get latest crowdlevel and status for each place
#     latest_crowd = CrowdData.objects.filter(
#         place=OuterRef('pk')
#     ).order_by('-timestamp')

#     places = Place.objects.filter(
#         category__icontains=category_cleaned
#     ).distinct().annotate(
#         latest_crowdlevel=Subquery(latest_crowd.values('crowdlevel')[:1]),
#         latest_status=Subquery(latest_crowd.values('status')[:1])
#     ).order_by('-latest_crowdlevel')

#     places_data = [{
#         'id': p.id,
#         'name': p.name,
#         'latitude': p.latitude,
#         'longitude': p.longitude,
#         'description': p.description,
#         'category': p.category,
#         'crowdlevel': p.latest_crowdlevel or 0,
#         'status': p.latest_status or 'Unknown',
#         'tags': [tag.name for tag in p.tags.all()],
#     } for p in places]

#     return JsonResponse({'places': places_data})

@api_view(['GET'])
def places_by_category(request, category):
    places = Place.objects.filter(category__iexact=category)
    places_data = []
    for place in places:
        crowdlevel = predict_crowd_for_place(place)
        places_data.append({
            'id': place.id,
            'name': place.name,
            'description': place.description,
            'category': place.category,
            'crowdlevel': crowdlevel,
            'district': place.district,
        })
    return JsonResponse({'places': places_data})

@api_view(['GET'])
def get_crowd_data(request,place_name):
    place = Place.objects.filter(name__iexact=place_name).first()
    crowd_data = CrowdData.objects.filter(place = place).order_by('-timestamp')[:5]
    """this shows the last 5 places where user have visited"""
    serializer = CrowdDataSerializer(crowd_data, many = True)
    return Response({"place" : PlaceSerializer(place).data, "crowd_data": serializer.data})

def generated_fake_crowd_data():
    places = Place.objects.all()
    
    for place in places:
        crowd_level = random.randint(0, 100)
        
        if crowd_level > 70:
            status = 'High'
        elif crowd_level > 30:
            status = 'Medium'
        else:
            status = 'Low'

        CrowdData.objects.create(
            place=place,
            crowdlevel=crowd_level,
            status=status,
        )


def weather_view(request):
    lat, lon = 27.7172, 85.3240
    weather_data = get_weather(lat, lon)

    return render(request, 'weather.html',{'weather': weather_data})

"""Content based filtering algorithm"""


TAG_WEIGHTS = {
    'spiritual': 1.5,
    'heritage': 1.2,
    'temple': 1.3,
    'nature': 1.1,
}

def recommend_places(request, place_name):
    """Recommend places based on the given place"""
    try:
        # Get the reference place
        reference_place = Place.objects.get(name__iexact=place_name)
        
        # Get tags of the reference place
        reference_tags = set(reference_place.tags.values_list('name', flat=True))
        
        # Find places with similar tags
        similar_places = []
        all_places = Place.objects.exclude(id=reference_place.id)
        
        for place in all_places:
            place_tags = set(place.tags.values_list('name', flat=True))
            
            # Calculate similarity score
            if reference_tags and place_tags:
                intersection = reference_tags.intersection(place_tags)
                union = reference_tags.union(place_tags)
                similarity = len(intersection) / len(union) if union else 0
                
                # Apply tag weights
                weighted_similarity = 0
                for tag in intersection:
                    weighted_similarity += TAG_WEIGHTS.get(tag.lower(), 1.0)
                
                if similarity > 0:  # Only include places with some similarity
                    similar_places.append({
                        'place': place,
                        'similarity_score': weighted_similarity * similarity
                    })
        
        # Sort by similarity score
        similar_places.sort(key=lambda x: x['similarity_score'], reverse=True)
        
        # Get top 5 recommendations
        recommendations = similar_places[:5]
        
        # Format response
        places_data = []
        for rec in recommendations:
            place = rec['place']
            latest_crowd = CrowdData.objects.filter(place=place).order_by('-timestamp').first()
            
            places_data.append({
                'id': place.id,
                'name': place.name,
                'description': place.description,
                'category': place.category,
                'similarity_score': round(rec['similarity_score'], 2),
                'crowdlevel': latest_crowd.crowdlevel if latest_crowd else 'N/A',
                'status': latest_crowd.status if latest_crowd else 'N/A',
                'tags': [tag.name for tag in place.tags.all()],
                'latitude': place.latitude,
                'longitude': place.longitude,
                'image': request.build_absolute_uri(place.image.url) if place.image else None
            })
        
        return JsonResponse({
            'reference_place': place_name,
            'recommendations': places_data
        })
        
    except Place.DoesNotExist:
        return JsonResponse({
            'error': f'Place "{place_name}" not found.'
        }, status=404)
    except Exception as e:
        return JsonResponse({
            'error': f'An error occurred: {str(e)}'
        }, status=500)

@api_view(['GET'])
def generate_fake_data(request):
    generated_fake_crowd_data()
    return Response({"message": "Fake data generated!"})


def place_details(request, place_id):
    place = get_object_or_404(Place, id=place_id)
    # Fetch other related data like crowd data if needed
    crowd_data = CrowdData.objects.filter(place=place).order_by('-timestamp').first()
    
    return render(request, 'placedetails.html', {
        'place': place,
        'crowd_data': crowd_data
    })

def place_list(request):
    places = Place.objects.all()
    return render(request, 'place_list.html', {'places': places})

@csrf_exempt
@login_required  # Ensure the user is authenticated
def save_user_location(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            latitude = data.get('latitude')
            longitude = data.get('longitude')
            
            if latitude is not None and longitude is not None:
                # Save or update user location
                user_location, created = UserLocation.objects.get_or_create(
                    user=request.user,
                    defaults={'latitude': latitude, 'longitude': longitude}
                )
                
                if not created:
                    # Update existing location
                    user_location.latitude = latitude
                    user_location.longitude = longitude
                    user_location.save()
                
                return JsonResponse({
                    'status': 'success',
                    'message': 'Location saved successfully'
                })
            else:
                return JsonResponse({
                    'status': 'error',
                    'message': 'Latitude and longitude are required'
                }, status=400)
                
        except json.JSONDecodeError:
            return JsonResponse({
                'status': 'error',
                'message': 'Invalid JSON data'
            }, status=400)
        except Exception as e:
            return JsonResponse({
                'status': 'error',
                'message': f'An error occurred: {str(e)}'
            }, status=500)
    
    return JsonResponse({
        'status': 'error',
        'message': 'Only POST method is allowed'
    }, status=405)

@api_view(['GET'])
@login_required  # Ensure the user is authenticated
def get_user_location(request):
    try:
        user_location = UserLocation.objects.get(user=request.user)
        return JsonResponse({
            'latitude': user_location.latitude,
            'longitude': user_location.longitude,
            'last_updated': user_location.created_at.isoformat()
        })
    except UserLocation.DoesNotExist:
        return JsonResponse({
            'error': 'No location data found for this user'
        }, status=404)

@api_view(['GET'])
def search_places(request):
    query = request.GET.get('q', '')  # Get the search query (place name)
    
    if not query:
        return JsonResponse({'error': 'No search term provided.'}, status=400)

    # Record search history if user is authenticated
    if request.user.is_authenticated:
        SearchHistory.objects.create(
            user=request.user,
            search_query=query,
            search_type='place'
        )

    places = Place.objects.filter(name__icontains=query)
    
    if not places:
        return JsonResponse({'error': 'No places found matching the query.'}, status=404)
    
    places_data = []
    
    for place in places:
        latest_crowd_data = CrowdData.objects.filter(place=place).aggregate(max_timestamp=Max('timestamp'))
        crowd_data = CrowdData.objects.filter(place=place, timestamp=latest_crowd_data['max_timestamp']).first()
        tags = list(place.tags.values_list('name', flat=True))

        places_data.append({
            'id': place.id,
            'name': place.name,
            'latitude': place.latitude,
            'longitude': place.longitude,
            'description': place.description,
            'popular_for': place.popular_for,
            'category': place.category,
            'crowd_level': crowd_data.crowdlevel if crowd_data else 'N/A',
            'tags': tags,
            'image': request.build_absolute_uri(place.image.url) if place.image else None
        })
    
    return JsonResponse({'places': places_data}, safe=False)

@api_view(['GET'])
def places_by_district(request, district_name):
    places = Place.objects.filter(district__iexact=district_name)
    places_data = []
    for place in places:
        crowdlevel = predict_crowd_for_place(place)
        places_data.append({
            'id': place.id,
            'name': place.name,
            'description': place.description,
            'category': place.category,
            'crowdlevel': crowdlevel,
            'district': place.district,
        })
    return JsonResponse({'places': places_data})


@api_view(['GET'])
def places_by_tag(request, tag_name):
    # Fetch places matching the tag
    places = Place.objects.filter(tags__name__iexact=tag_name)
    result = []

    for place in places:
        latest_crowd = CrowdData.objects.filter(place=place).order_by('-timestamp').first()

        result.append({
            'id': place.id,
            'name': place.name,
            'description': place.description,
            'popular_for': place.popular_for,
            'category': place.category,
            'crowdlevel': latest_crowd.crowdlevel if latest_crowd else 'N/A',
            'status': latest_crowd.status if latest_crowd else 'N/A',
            'tags': list(place.tags.values_list('name', flat=True)),
            'lat': place.latitude,
            'lng': place.longitude,
            'image': request.build_absolute_uri(place.image.url) if place.image else None
        })

    return JsonResponse({'places': result})

@login_required
def add_place(request):
    from .models import Tag
    tags = Tag.objects.all()
    if request.method == 'POST':
        try:
            # Get form data
            name = request.POST.get('name')
            description = request.POST.get('description')
            popular_for = request.POST.get('popular_for')
            category = request.POST.get('category')
            district = request.POST.get('district')
            location = request.POST.get('location')
            latitude = request.POST.get('latitude')
            longitude = request.POST.get('longitude')
            image = request.FILES.get('image')
            tags = request.POST.getlist('tags')
            crowdlevel = request.POST.get('crowdlevel')

            # Debug print
            print(f"Received data: {request.POST}")
            print(f"Received files: {request.FILES}")

            # Validate required fields
            if not all([name, description, popular_for, category, district, location, latitude, longitude, image, crowdlevel]):
                missing_fields = []
                if not name: missing_fields.append('name')
                if not description: missing_fields.append('description')
                if not popular_for: missing_fields.append('popular_for')
                if not category: missing_fields.append('category')
                if not district: missing_fields.append('district')
                if not location: missing_fields.append('location')
                if not latitude: missing_fields.append('latitude')
                if not longitude: missing_fields.append('longitude')
                if not image: missing_fields.append('image')
                if not crowdlevel: missing_fields.append('crowdlevel')
                
                messages.error(request, f'Missing required fields: {", ".join(missing_fields)}')
                return render(request, 'addplace.html', {'tags': tags})

            # Set status based on crowdlevel
            crowdlevel = int(crowdlevel)
            if crowdlevel > 70:
                status = 'High'
            elif crowdlevel > 30:
                status = 'Medium'
            else:
                status = 'Low'

            # Create new place with approval status based on user type
            place = Place.objects.create(
                name=name,
                description=description,
                popular_for=popular_for,
                category=category,
                district=district,
                location=location,
                latitude=latitude,
                longitude=longitude,
                image=image,
                added_by=request.user,
                is_approved=request.user.is_superuser  # Auto-approve if superuser
            )

            # Add tags
            for tag_name in tags:
                tag_name = tag_name.strip()
                if tag_name:
                    tag, created = Tag.objects.get_or_create(name=tag_name)
                    place.tags.add(tag)

            # Create crowd data
            CrowdData.objects.create(
                place=place,
                crowdlevel=crowdlevel,
                status=status
            )

            if request.user.is_superuser:
                messages.success(request, 'Place added successfully!')
                return redirect('place_details', place_id=place.id)
            else:
                messages.success(request, 'Place submitted for approval! An admin will review it soon.')
            return redirect('place_details', place_id=place.id)
        except Exception as e:
            print(f"Error adding place: {str(e)}")  # Debug print
            messages.error(request, f'Error adding place: {str(e)}')
            return render(request, 'addplace.html', {'tags': tags})
    
    return render(request, 'addplace.html', {'tags': tags})

@login_required
def update_profile(request):
    if request.method == 'POST':
        user = request.user
        username = request.POST.get('username')
        email = request.POST.get('email')
        new_password = request.POST.get('new_password')
        confirm_password = request.POST.get('confirm_password')

        # Basic validation
        if not username or not email:
            messages.error(request, "Username and email are required.")
            return redirect('profile_view')

        # Check if username is taken by another user
        if User.objects.filter(username=username).exclude(pk=user.pk).exists():
            messages.error(request, "This username is already taken.")
            return redirect('profile_view')

        user.username = username
        user.email = email

        password_changed = False
        if new_password:
            if new_password != confirm_password:
                messages.error(request, "Passwords do not match.")
                return redirect('profile_view')
            user.set_password(new_password)
            password_changed = True
            messages.success(request, "Password updated successfully.")

        user.save()

        # Re-authenticate and log in the user if password was changed
        if password_changed:
            user = authenticate(request, username=username, password=new_password)
            if user is not None:
                login(request, user)

        messages.success(request, "Profile updated successfully.")
        return redirect('map_view')  # Redirect to map_view (home page)

    # On GET request, just show the profile page.
    return redirect('profile_view')

@login_required
def add_place_form(request):
    return render(request, 'addplace.html')

@api_view(['GET'])
@login_required
def get_search_history(request):
    user = request.user
    search_history = SearchHistory.objects.filter(user=user).order_by('-timestamp')[:10]
    
    history_data = [{
        'search_query': item.search_query,
        'search_type': item.search_type,
        'timestamp': item.timestamp.strftime('%Y-%m-%d %H:%M:%S')
    } for item in search_history]
    
    return JsonResponse({'searches': history_data})

def haversine(lat1, lon1, lat2, lon2):
    """Calculate the great circle distance between two points on the earth"""
    # Convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    r = 6371  # Radius of earth in kilometers
    return c * r

@login_required
def recommended_places_nearby(request):
    try:
        # Get user's current location
        user_location = UserLocation.objects.get(user=request.user)
        user_lat = user_location.latitude
        user_lon = user_location.longitude
        
        # Get all places
        places = Place.objects.all()
        nearby_places = []
        
        for place in places:
            if place.latitude and place.longitude:
                distance = haversine(user_lat, user_lon, place.latitude, place.longitude)
                
                # Only include places within 10km
                if distance <= 10:
                    latest_crowd = CrowdData.objects.filter(place=place).order_by('-timestamp').first()
                    
                    nearby_places.append({
                        'id': place.id,
                        'name': place.name,
                        'description': place.description,
                        'category': place.category,
                        'distance': round(distance, 2),
                        'crowdlevel': latest_crowd.crowdlevel if latest_crowd else 'N/A',
                        'status': latest_crowd.status if latest_crowd else 'N/A',
                        'tags': [tag.name for tag in place.tags.all()],
                        'latitude': place.latitude,
                        'longitude': place.longitude
                    })
        
        # Sort by distance
        nearby_places.sort(key=lambda x: x['distance'])
        
        return JsonResponse({'nearby_places': nearby_places[:10]})  # Return top 10
        
    except UserLocation.DoesNotExist:
        return JsonResponse({'error': 'User location not found'}, status=404)
    except Exception as e:
        return JsonResponse({'error': f'An error occurred: {str(e)}'}, status=500)

@login_required
def delete_place(request, place_id):
    place = get_object_or_404(Place, id=place_id)

    # Check for permission
    if not (request.user.is_superuser or place.added_by == request.user):
        messages.error(request, "You don't have permission to delete this place.")
        if request.headers.get('x-requested-with') == 'XMLHttpRequest':
            return JsonResponse({'success': False, 'error': "You don't have permission."}, status=403)
        return redirect('place_details', place_id=place_id)

    if request.method == 'POST':
        try:
            place_name = place.name
            place.delete()
            messages.success(request, f'Place "{place_name}" has been deleted successfully.')

            if request.headers.get('x-requested-with') == 'XMLHttpRequest':
                # For AJAX, return success and a redirect URL
                return JsonResponse({
                    'success': True,
                    'message': f'Place "{place_name}" has been deleted successfully.',
                    'redirect_url': reverse('home') 
                })
            
            # For standard form submissions, redirect to home
            return redirect('home')

        except Exception as e:
            error_message = f'An error occurred: {e}'
            messages.error(request, error_message)
            if request.headers.get('x-requested-with') == 'XMLHttpRequest':
                return JsonResponse({'success': False, 'error': str(e)}, status=500)
            return redirect('place_details', place_id=place_id)

    # If it's a GET request, redirect away, as we no longer have a confirmation page.
    # This prevents the TemplateDoesNotExist error.
    return redirect('place_details', place_id=place_id)

def save_place(request, place_id):
    place = get_object_or_404(Place, id=place_id)
    
    if request.method == 'POST':
        # Update place details
        place.name = request.POST.get('name', place.name)
        place.description = request.POST.get('description', place.description)
        place.popular_for = request.POST.get('popular_for', place.popular_for)
        place.category = request.POST.get('category', place.category)
        place.district = request.POST.get('district', place.district)
        place.location = request.POST.get('location', place.location)
        
        # Update coordinates if provided
        latitude = request.POST.get('latitude')
        longitude = request.POST.get('longitude')
        if latitude and longitude:
            place.latitude = float(latitude)
            place.longitude = float(longitude)
        
        # Update image if provided
        if 'image' in request.FILES:
            place.image = request.FILES['image']
        
        place.save()
        
        # Update tags
        tags = request.POST.getlist('tags')
        place.tags.clear()
        for tag_name in tags:
            if tag_name.strip():
                tag, created = Tag.objects.get_or_create(name=tag_name.strip())
                place.tags.add(tag)
        
        messages.success(request, 'Place updated successfully!')
        return redirect('place_details', place_id=place_id)
    
    return render(request, 'edit_place.html', {'place': place})

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

@api_view(['POST'])
def predict_crowd(request):
    """
    Predict crowd level for a given place, date, and time slot
    """
    try:
        # Get parameters from request
        place_id = request.data.get('place_id')
        date_str = request.data.get('date')
        time_slot = request.data.get('time_slot')
        weather_condition = request.data.get('weather_condition', 'Sunny')
        
        if not all([place_id, date_str, time_slot]):
            return JsonResponse({
                'error': 'Missing required parameters: place_id, date, time_slot'
            }, status=400)
        
        # Get place details
        try:
            place = Place.objects.get(id=place_id)
        except Place.DoesNotExist:
            return JsonResponse({
                'error': f'Place with ID {place_id} not found'
            }, status=404)
        
        # Parse date
        try:
            date = datetime.strptime(date_str, '%Y-%m-%d')
        except ValueError:
            return JsonResponse({
                'error': 'Invalid date format. Use YYYY-MM-DD'
            }, status=400)
        
        # Load the trained model
        model_path = 'crowd_prediction_model.pkl'
        if not os.path.exists(model_path):
            return JsonResponse({
                'error': 'Trained model not found. Please train the model first.'
            }, status=500)
        
        model_data = joblib.load(model_path)
        model = model_data['model']
        encoders = model_data['label_encoders']
        
        # Prepare features
        day_of_week = date.weekday()
        month = date.month
        season = get_season(month)
        is_weekend = 1 if day_of_week >= 5 else 0
        is_holiday = 0  # You can add holiday logic here
        
        # Encode categorical features
        features = [
            place_id,
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
        
        return JsonResponse({
            'place_id': place_id,
            'place_name': place.name,
            'date': date_str,
            'time_slot': time_slot,
            'weather_condition': weather_condition,
            'predicted_crowd_level': predicted_crowd,
            'status': status,
            'features_used': {
                'category': place.category,
                'district': place.district,
                'day_of_week': day_of_week,
                'month': month,
                'season': season,
                'is_weekend': is_weekend,
                'is_holiday': is_holiday
            }
        })
        
    except Exception as e:
        return JsonResponse({
            'error': f'Prediction failed: {str(e)}'
        }, status=500)

def register_view(request):
    return render(request, 'register.html')

def predict_crowd_for_place(place, time_slot='morning'):
    model_path = 'crowd_prediction_model.pkl'
    if not os.path.exists(model_path):
        return 0
    model_data = joblib.load(model_path)
    model = model_data['model']
    encoders = model_data['label_encoders']
    now = datetime.now()
    day_of_week = now.weekday()
    month = now.month
    season = get_current_season()
    is_weekend = 1 if day_of_week >= 5 else 0
    is_holiday = 0
    weather = get_current_weather()
    try:
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
            encoders['weather_condition'].transform([weather])[0]
        ]
        predicted_crowd = model.predict([features])[0]
        return float(max(0, min(100, round(predicted_crowd, 1))))
    except Exception as e:
        return 0