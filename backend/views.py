import random
import json
from django.utils.timezone import now, timezone
from django.shortcuts import render, get_object_or_404,redirect
from django.http import JsonResponse
from rest_framework.response import Response
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated
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
import logging
from rest_framework import status
import sys
sys.path.append(os.path.join(os.path.dirname(__file__)))

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
    if request.user.is_superuser:
        recommended_places = Place.objects.filter(tags__in=preferred_tags).distinct()
    else:
        recommended_places = Place.objects.filter(tags__in=preferred_tags, is_approved=True).distinct()

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
    time_slot = request.GET.get('time_slot')
    hour = request.GET.get('hour')
    if not time_slot:
        now = datetime.now()
        hour = now.hour
        if 5 <= hour < 12:
            time_slot = 'morning'
        elif 12 <= hour < 17:
            time_slot = 'afternoon'
        elif 17 <= hour < 21:
            time_slot = 'evening'
        else:
            time_slot = 'morning'
    # Save search history
    save_search_history(request.user, category, 'category')
    
    # Filter places based on user permissions
    if request.user.is_superuser:
        places = Place.objects.filter(category__iexact=category)
    else:
        places = Place.objects.filter(category__iexact=category, is_approved=True)
    
    places_data = []
    for place in places:
        crowdlevel = predict_crowd_for_place(place, time_slot)
        places_data.append({
            'id': place.id,
            'name': place.name,
            'description': place.description,
            'category': place.category,
            'crowdlevel': crowdlevel,
            'district': place.district,
            'latitude': place.latitude,
            'longitude': place.longitude,
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
        if request.user.is_superuser:
            all_places = Place.objects.exclude(id=reference_place.id)
        else:
            all_places = Place.objects.exclude(id=reference_place.id).filter(is_approved=True)
        
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
    
    # Check if user can view this place
    can_view = (
        request.user.is_superuser or  # Superusers can view all places
        place.added_by == request.user or  # Place owner can view their own place
        place.is_approved  # Approved places are visible to everyone
    )
    
    if not can_view:
        messages.warning(request, 'This place is pending admin approval and is not yet visible to the public.')
        return redirect('home')  # Redirect to home page
    
    # Use the improved model for prediction (same as bar graph)
    from datetime import datetime
    from backend.improved_ml_model import ImprovedCrowdPredictionModel
    import os
    crowdlevel = None
    improved_model_path = 'improved_crowd_prediction_model.pkl'
    if os.path.exists(improved_model_path):
        model = ImprovedCrowdPredictionModel()
        if model.load_model():
            now = datetime.now()
            day_of_week = now.weekday()
            month = now.month
            season = get_current_season()
            is_weekend = 1 if day_of_week >= 5 else 0
            is_holiday = 0
            # Use default hour for details page (e.g., 14 for afternoon)
            hour = 14
            crowdlevel = model.predict(
                place_id=place.id,
                category=place.category,
                district=place.district,
                time_slot='afternoon',
                day_of_week=day_of_week,
                month=month,
                season=season,
                is_weekend=is_weekend,
                is_holiday=is_holiday,
                weather_condition='Sunny',
                hour=hour
            )
    # Fallback: use latest crowd data if improved model not available
    if crowdlevel is None:
        crowd_data = CrowdData.objects.filter(place=place).order_by('-timestamp').first()
        crowdlevel = crowd_data.crowdlevel if crowd_data else 'N/A'
    today = timezone.now().date().isoformat()  # e.g., '2025-06-27'
    closed_list = []
    if place.closed_dates:
        closed_list = [d.strip() for d in place.closed_dates.split(',') if d.strip()]
    return render(request, 'placedetails.html', {
        'place': place,
        'crowdlevel': crowdlevel,
        'today': today,
        'closed_list': closed_list,
    })

def place_list(request):
    if request.user.is_superuser:
        places = Place.objects.all()
    else:
        places = Place.objects.filter(is_approved=True)
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

MODEL_CACHE = {}

@api_view(['GET'])
def search_places(request):
    query = request.GET.get('q', '')  # Get the search query (place name)
    
    if not query:
        return JsonResponse({'error': 'No search term provided.'}, status=400)

    # Record search history if user is authenticated
    save_search_history(request.user, query, 'place')

    # Filter places based on user permissions
    if request.user.is_superuser:
        places = Place.objects.filter(name__icontains=query)
    else:
        places = Place.objects.filter(name__icontains=query, is_approved=True)
    
    if not places:
        return JsonResponse({'error': 'No places found matching the query.'}, status=404)
    
    places_data = []
    
    for place in places:
        latest_crowd_data = CrowdData.objects.filter(place=place).aggregate(max_timestamp=Max('timestamp'))
        crowd_data = CrowdData.objects.filter(place=place, timestamp=latest_crowd_data['max_timestamp']).first()
        tags = list(place.tags.values_list('name', flat=True))
        crowd_pattern = get_crowd_pattern_from_db(place)
        places_data.append({
            'id': place.id,
            'name': place.name,
            'latitude': place.latitude,
            'longitude': place.longitude,
            'description': place.description,
            'popular_for': place.popular_for,
            'category': place.category,
            'crowdlevel': crowd_data.crowdlevel if crowd_data else 'N/A',
            'crowd_pattern': crowd_pattern,
            'tags': tags,
            'image': request.build_absolute_uri(place.image.url) if place.image else None
        })
    print(f"[DEBUG] search_places API response: {places_data}")
    return JsonResponse({'places': places_data}, safe=False)

@api_view(['GET'])
def places_by_district(request, district_name):
    time_slot = request.GET.get('time_slot')
    hour = request.GET.get('hour')
    if not time_slot:
        now = datetime.now()
        hour = now.hour
        if 5 <= hour < 12:
            time_slot = 'morning'
        elif 12 <= hour < 17:
            time_slot = 'afternoon'
        elif 17 <= hour < 21:
            time_slot = 'evening'
        else:
            time_slot = 'morning'
    save_search_history(request.user, district_name, 'district')
    
    # Filter places based on user permissions
    if request.user.is_superuser:
        places = Place.objects.filter(district__iexact=district_name)
    else:
        places = Place.objects.filter(district__iexact=district_name, is_approved=True)
    
    print(f"Found {places.count()} places for district {district_name}")
    places_data = []
    for place in places:
        crowdlevel = predict_crowd_for_place(place, time_slot)
        place_data = {
            'id': place.id,
            'name': place.name,
            'description': place.description,
            'category': place.category,
            'crowdlevel': crowdlevel,
            'district': place.district,
            'latitude': place.latitude,
            'longitude': place.longitude,
        }
        print(f"Place data: {place_data}")
        places_data.append(place_data)
    print(f"Sending {len(places_data)} places")
    return JsonResponse({'places': places_data})


@api_view(['GET'])
def places_by_tag(request, tag_name):
    # Fetch places matching the tag
    if request.user.is_superuser:
        places = Place.objects.filter(tags__name__iexact=tag_name)
    else:
        places = Place.objects.filter(tags__name__iexact=tag_name, is_approved=True)
    
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
            opening_time = request.POST.get('opening_time')
            closing_time = request.POST.get('closing_time')
            best_time_to_visit = request.POST.get('best_time_to_visit')
            closed_dates = request.POST.get('closed_dates')
            
            # Entry fee fields
            has_entry_fee = request.POST.get('has_entry_fee') == 'on'
            tourist_fee_npr = request.POST.get('tourist_fee_npr')
            tourist_fee_usd = request.POST.get('tourist_fee_usd')
            saarc_fee_npr = request.POST.get('saarc_fee_npr')
            local_fee_npr = request.POST.get('local_fee_npr')
            fee_description = request.POST.get('fee_description')

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
                is_approved=request.user.is_superuser,  # Auto-approve if superuser
                has_entry_fee=has_entry_fee,
                tourist_fee_npr=tourist_fee_npr if tourist_fee_npr else None,
                tourist_fee_usd=tourist_fee_usd if tourist_fee_usd else None,
                saarc_fee_npr=saarc_fee_npr if saarc_fee_npr else None,
                local_fee_npr=local_fee_npr if local_fee_npr else None,
                fee_description=fee_description if fee_description else '',
                opening_time=opening_time if opening_time else None,
                closing_time=closing_time if closing_time else None,
                best_time_to_visit=best_time_to_visit if best_time_to_visit else '',
                closed_dates=closed_dates if closed_dates else ''
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
                messages.success(request, 'Place submitted for approval! An admin will review it soon. You can view your place while it\'s pending approval.')
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
        if request.user.is_superuser:
            places = Place.objects.all()
        else:
            places = Place.objects.filter(is_approved=True)
        
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
            place_district = place.district  # Store district before deletion
            place.delete()
            messages.success(request, f'Place "{place_name}" has been deleted successfully.')

            if request.headers.get('x-requested-with') == 'XMLHttpRequest':
                # For AJAX, return success and a redirect URL to home page
                return JsonResponse({
                    'success': True,
                    'message': f'Place "{place_name}" has been deleted successfully.',
                    'redirect_url': '/'  # Redirect to home page
                })
            
            # For standard form submissions, redirect to home page
            return redirect('/')

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
    tags = Tag.objects.all()
    place_tag_names = list(place.tags.values_list('name', flat=True))
    
    if request.method == 'POST':
        # Update place details
        place.name = request.POST.get('name', place.name)
        place.description = request.POST.get('description', place.description)
        place.popular_for = request.POST.get('popular_for', place.popular_for)
        place.category = request.POST.get('category', place.category)
        place.district = request.POST.get('district', place.district)
        place.location = request.POST.get('location', place.location)
        
        # Update entry fee fields
        place.has_entry_fee = request.POST.get('has_entry_fee') == 'on'
        place.tourist_fee_npr = request.POST.get('tourist_fee_npr') if request.POST.get('tourist_fee_npr') else None
        place.tourist_fee_usd = request.POST.get('tourist_fee_usd') if request.POST.get('tourist_fee_usd') else None
        place.saarc_fee_npr = request.POST.get('saarc_fee_npr') if request.POST.get('saarc_fee_npr') else None
        place.local_fee_npr = request.POST.get('local_fee_npr') if request.POST.get('local_fee_npr') else None
        place.fee_description = request.POST.get('fee_description', '')
        
        # Update opening hours fields
        opening_time = request.POST.get('opening_time')
        closing_time = request.POST.get('closing_time')
        if opening_time:
            place.opening_time = opening_time
        else:
            place.opening_time = None
        if closing_time:
            place.closing_time = closing_time
        else:
            place.closing_time = None
        place.best_time_to_visit = request.POST.get('best_time_to_visit', '')
        place.closed_dates = request.POST.get('closed_dates', '')
        
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
    
    return render(request, 'edit_place.html', {'place': place, 'tags': tags, 'place_tag_names': place_tag_names})

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
        
        # Try to use improved model first
        improved_model_path = 'improved_crowd_prediction_model.pkl'
        if os.path.exists(improved_model_path):
            try:
                from backend.improved_ml_model import ImprovedCrowdPredictionModel
                model = ImprovedCrowdPredictionModel()
                if model.load_model():
                    # Prepare features for improved model
                    day_of_week = date.weekday()
                    month = date.month
                    season = get_season(month)
                    is_weekend = 1 if day_of_week >= 5 else 0
                    is_holiday = 0
                    
                    # Get hour for time slot
                    if time_slot == 'morning':
                        hour = 9
                    elif time_slot == 'afternoon':
                        hour = 14
                    else:  # evening
                        hour = 19
                    
                    predicted_crowd = model.predict(
                        place_id=place_id,
                        category=place.category,
                        district=place.district,
                        time_slot=time_slot,
                        day_of_week=day_of_week,
                        month=month,
                        season=season,
                        is_weekend=is_weekend,
                        is_holiday=is_holiday,
                        weather_condition=weather_condition,
                        hour=hour
                    )
                    
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
                        'model_used': 'improved',
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
                print(f"[DEBUG] Improved model prediction failed: {e}")
                # Continue to fallback model
        
        # Fallback to old model
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
            'model_used': 'legacy',
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
    """Predict crowd level using the improved model"""
    try:
        # Try to load the improved model first
        improved_model_path = 'improved_crowd_prediction_model.pkl'
        if os.path.exists(improved_model_path):
            # Use improved model
            from backend.improved_ml_model import ImprovedCrowdPredictionModel
            model = ImprovedCrowdPredictionModel()
            if model.load_model():
                now = datetime.now()
                day_of_week = now.weekday()
                month = now.month
                season = get_current_season()
                is_weekend = 1 if day_of_week >= 5 else 0
                is_holiday = 0
                weather = get_current_weather()
                
                # Get hour for time slot
                if time_slot == 'morning':
                    hour = 9
                elif time_slot == 'afternoon':
                    hour = 14
                else:  # evening
                    hour = 19
                
                predicted_crowd = model.predict(
                    place_id=place.id,
                    category=place.category,
                    district=place.district,
                    time_slot=time_slot,
                    day_of_week=day_of_week,
                    month=month,
                    season=season,
                    is_weekend=is_weekend,
                    is_holiday=is_holiday,
                    weather_condition=weather,
                    hour=hour
                )
                return float(predicted_crowd)
        
        # Fallback to old model if improved model not available
        model_path = 'crowd_prediction_model.pkl'
        if model_path not in MODEL_CACHE:
            if not os.path.exists(model_path):
                return 0
            model_data = joblib.load(model_path)
            MODEL_CACHE[model_path] = model_data
        else:
            model_data = MODEL_CACHE[model_path]
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
            print(f"[DEBUG] Model prediction error for place {place.name}: {e}")
            return 0
    except Exception as e:
        print(f"[DEBUG] Improved model prediction error for place {place.name}: {e}")
        return 0

# Utility to save search history
def save_search_history(user, query, search_type):
    if user.is_authenticated:
        SearchHistory.objects.create(
            user=user,
            search_query=query,
            search_type=search_type
        )

def get_crowd_pattern_from_db(place):
    pattern = []
    for hour in range(24):
        cp = CrowdPattern.objects.filter(place=place, hour=hour).first()
        crowd = cp.crowdlevel if cp else 0
        pattern.append({'hour': hour, 'crowdlevel': crowd})
    return pattern

@api_view(["PUT", "PATCH"])
@permission_classes([IsAuthenticated])
def api_update_place(request, place_id):
    try:
        place = get_object_or_404(Place, id=place_id)
        # Only allow the user who added the place or a superuser to update
        if not (request.user.is_superuser or place.added_by == request.user):
            return Response({"error": "You do not have permission to update this place."}, status=status.HTTP_403_FORBIDDEN)

        print(f"[DEBUG] Updating place {place_id} with data: {request.data}")
        print(f"[DEBUG] Content type: {request.content_type}")
        print(f"[DEBUG] Files: {request.FILES}")
        
        # Handle both JSON and multipart form data
        data = request.data.copy()
        
        # If there are files, merge them with the data
        if request.FILES:
            for key, file in request.FILES.items():
                data[key] = file
                print(f"[DEBUG] Added file {key}: {file.name}")
        
        serializer = PlaceSerializer(place, data=data, partial=True)
        if serializer.is_valid():
            serializer.save()
            print(f"[DEBUG] Place {place_id} updated successfully")
            return Response(serializer.data, status=status.HTTP_200_OK)
        else:
            print(f"[DEBUG] Serializer errors: {serializer.errors}")
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    except Exception as e:
        print(f"[DEBUG] Error updating place {place_id}: {str(e)}")
        return Response({"error": f"An error occurred: {str(e)}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

@api_view(['GET'])
def improved_crowd_predictions(request):
    """
    Get improved crowd predictions using the enhanced model with Nepal-specific behavioral patterns
    """
    try:
        # Get query parameters
        district = request.GET.get('district')
        category = request.GET.get('category')
        time_slot = request.GET.get('time_slot')
        weather = request.GET.get('weather', 'Sunny')
        
        # Determine current time slot if not provided
        if not time_slot:
            now = datetime.now()
            hour = now.hour
            if 5 <= hour < 12:
                time_slot = 'morning'
            elif 12 <= hour < 17:
                time_slot = 'afternoon'
            elif 17 <= hour < 21:
                time_slot = 'evening'
            else:
                time_slot = 'morning'
        
        # Filter places based on parameters
        places = Place.objects.all()
        if district:
            places = places.filter(district__iexact=district)
        if category:
            places = places.filter(category__iexact=category)
        
        # Filter by approval status based on user permissions
        if not request.user.is_superuser:
            places = places.filter(is_approved=True)
        
        if not places.exists():
            return JsonResponse({
                'error': 'No places found for the given criteria'
            }, status=404)
        
        # Try to use improved model
        improved_model_path = 'improved_crowd_prediction_model.pkl'
        if not os.path.exists(improved_model_path):
            return JsonResponse({
                'error': 'Improved model not found. Please train the improved model first.'
            }, status=500)
        
        try:
            from backend.improved_ml_model import ImprovedCrowdPredictionModel
            model = ImprovedCrowdPredictionModel()
            if not model.load_model():
                return JsonResponse({
                    'error': 'Failed to load improved model.'
                }, status=500)
            
            # Get current context
            now = datetime.now()
            day_of_week = now.weekday()
            month = now.month
            season = get_current_season()
            is_weekend = 1 if day_of_week >= 5 else 0
            is_holiday = 0
            
            # Get hour for time slot
            if time_slot == 'morning':
                hour = 9
            elif time_slot == 'afternoon':
                hour = 14
            else:  # evening
                hour = 19
            
            # Get predictions for all places
            places_data = []
            for place in places:
                try:
                    predicted_crowd = model.predict(
                        place_id=place.id,
                        category=place.category,
                        district=place.district,
                        time_slot=time_slot,
                        day_of_week=day_of_week,
                        month=month,
                        season=season,
                        is_weekend=is_weekend,
                        is_holiday=is_holiday,
                        weather_condition=weather,
                        hour=hour
                    )
                    
                    # Determine crowd status
                    if predicted_crowd > 70:
                        status = 'High'
                        color = 'red'
                    elif predicted_crowd > 30:
                        status = 'Medium'
                        color = 'orange'
                    else:
                        status = 'Low'
                        color = 'green'
                    
                    places_data.append({
                        'id': place.id,
                        'name': place.name,
                        'description': place.description,
                        'category': place.category,
                        'district': place.district,
                        'latitude': place.latitude,
                        'longitude': place.longitude,
                        'crowdlevel': predicted_crowd,
                        'status': status,
                        'color': color,
                        'time_slot': time_slot,
                        'weather': weather,
                        'is_weekend': is_weekend,
                        'season': season
                    })
                except Exception as e:
                    print(f"[DEBUG] Prediction error for place {place.name}: {e}")
                    continue
            
            # Sort by crowd level (highest first)
            places_data.sort(key=lambda x: x['crowdlevel'], reverse=True)
            
            return JsonResponse({
                'places': places_data,
                'model_used': 'improved',
                'total_places': len(places_data),
                'filters_applied': {
                    'district': district,
                    'category': category,
                    'time_slot': time_slot,
                    'weather': weather
                },
                'context': {
                    'current_time': now.strftime('%Y-%m-%d %H:%M:%S'),
                    'day_of_week': day_of_week,
                    'is_weekend': is_weekend,
                    'season': season
                }
            })
            
        except Exception as e:
            return JsonResponse({
                'error': f'Improved model prediction failed: {str(e)}'
            }, status=500)
            
    except Exception as e:
        return JsonResponse({
            'error': f'Request failed: {str(e)}'
        }, status=500)

@login_required
def pending_places(request):
    """View for admins to see places pending approval"""
    if not request.user.is_superuser:
        messages.error(request, "You don't have permission to view pending places.")
        return redirect('home')
    
    pending_places = Place.objects.filter(is_approved=False).order_by('-id')
    
    return render(request, 'pending_places.html', {
        'pending_places': pending_places,
        'total_pending': pending_places.count()
    })

@api_view(['POST'])
@login_required
def admin_approve_place(request, place_id):
    """API endpoint for admins to approve a place"""
    if not request.user.is_superuser:
        return JsonResponse({'success': False, 'error': 'Permission denied'}, status=403)
    
    try:
        place = get_object_or_404(Place, id=place_id)
        place.is_approved = True
        place.save()
        
        return JsonResponse({
            'success': True,
            'message': f'Place "{place.name}" has been approved successfully.'
        })
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': f'An error occurred: {str(e)}'
        }, status=500)

@api_view(['POST'])
@login_required
def admin_reject_place(request, place_id):
    """API endpoint for admins to reject a place"""
    if not request.user.is_superuser:
        return JsonResponse({'success': False, 'error': 'Permission denied'}, status=403)
    
    try:
        place = get_object_or_404(Place, id=place_id)
        place_name = place.name
        place.delete()
        
        return JsonResponse({
            'success': True,
            'message': f'Place "{place_name}" has been rejected and deleted.'
        })
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': f'An error occurred: {str(e)}'
        }, status=500)

@api_view(['GET'])
def tourism_crowd_data_for_charts(request):
    """
    Get tourism-based crowd data specifically for bar graph visualization
    This uses the trained tourism model to provide realistic crowd predictions
    """
    try:
        # Get query parameters
        district = request.GET.get('district')
        category = request.GET.get('category')
        time_slot = request.GET.get('time_slot')
        limit = int(request.GET.get('limit', 7))  # Number of places to show in chart (changed from 10 to 7)
        
        # Determine current time slot if not provided
        if not time_slot:
            now = datetime.now()
            hour = now.hour
            if 5 <= hour < 12:
                time_slot = 'morning'
            elif 12 <= hour < 17:
                time_slot = 'afternoon'
            elif 17 <= hour < 21:
                time_slot = 'evening'
            else:
                time_slot = 'morning'
        
        # Filter places based on parameters
        places = Place.objects.all()
        if district:
            places = places.filter(district__iexact=district)
        if category:
            places = places.filter(category__iexact=category)
        
        # Filter by approval status based on user permissions
        if not request.user.is_superuser:
            places = places.filter(is_approved=True)
        
        if not places.exists():
            return JsonResponse({
                'error': 'No places found for the given criteria'
            }, status=404)
        
        # Use the improved tourism model
        improved_model_path = 'improved_crowd_prediction_model.pkl'
        if not os.path.exists(improved_model_path):
            return JsonResponse({
                'error': 'Tourism model not found. Please train the model first.'
            }, status=500)
        
        try:
            from backend.improved_ml_model import ImprovedCrowdPredictionModel
            model = ImprovedCrowdPredictionModel()
            if not model.load_model():
                return JsonResponse({
                    'error': 'Failed to load tourism model.'
                }, status=500)
            
            # Get current context for tourism predictions
            now = datetime.now()
            day_of_week = now.weekday()
            month = now.month
            season = get_current_season()
            is_weekend = 1 if day_of_week >= 5 else 0
            is_holiday = 0
            
            # Get hour for time slot
            if time_slot == 'morning':
                hour = 9
            elif time_slot == 'afternoon':
                hour = 14
            else:  # evening
                hour = 19
            
            # Get predictions for all places
            places_data = []
            for place in places:
                try:
                    predicted_crowd = model.predict(
                        place_id=place.id,
                        category=place.category,
                        district=place.district,
                        time_slot=time_slot,
                        day_of_week=day_of_week,
                        month=month,
                        season=season,
                        is_weekend=is_weekend,
                        is_holiday=is_holiday,
                        weather_condition='Sunny',  # Default weather
                        hour=hour
                    )
                    
                    # Determine crowd status with updated color scheme: High-Red, Medium-Green, Low-Yellow
                    if predicted_crowd > 70:
                        status = 'High'
                        color = 'red'
                    elif predicted_crowd >= 30:
                        status = 'Medium'
                        color = 'green'
                    else:
                        status = 'Low'
                        color = 'yellow'
                    
                    places_data.append({
                        'id': place.id,
                        'name': place.name,
                        'description': place.description,
                        'category': place.category,
                        'district': place.district,
                        'latitude': place.latitude,
                        'longitude': place.longitude,
                        'crowdlevel': predicted_crowd,
                        'status': status,
                        'color': color,
                        'time_slot': time_slot,
                        'is_weekend': is_weekend,
                        'season': season
                    })
                except Exception as e:
                    print(f"[DEBUG] Tourism prediction error for place {place.name}: {e}")
                    continue
            
            # Sort by crowd level (highest first) and limit results
            places_data.sort(key=lambda x: x['crowdlevel'], reverse=True)
            places_data = places_data[:limit]
            
            # Prepare data specifically for bar charts
            chart_data = {
                'labels': [place['name'] for place in places_data],
                'crowdlevels': [place['crowdlevel'] for place in places_data],
                'colors': [place['color'] for place in places_data],
                'categories': [place['category'] for place in places_data],
                'districts': [place['district'] for place in places_data],
                'place_ids': [place['id'] for place in places_data]
            }
            
            return JsonResponse({
                'places': places_data,
                'chart_data': chart_data,
                'model_used': 'tourism_improved',
                'total_places': len(places_data),
                'filters_applied': {
                    'district': district,
                    'category': category,
                    'time_slot': time_slot
                },
                'context': {
                    'current_time': now.strftime('%Y-%m-%d %H:%M:%S'),
                    'day_of_week': day_of_week,
                    'is_weekend': is_weekend,
                    'season': season,
                    'tourist_season': 'Peak' if month in [10, 11, 4, 5] else 'Shoulder' if month in [3, 9, 12] else 'Low'
                }
            })
            
        except Exception as e:
            return JsonResponse({
                'error': f'Tourism model prediction failed: {str(e)}'
            }, status=500)
            
    except Exception as e:
        return JsonResponse({
            'error': f'Request failed: {str(e)}'
        }, status=500)