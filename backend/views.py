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


# Create your views here.

def home(request):
    return render(request, 'register.html')
    """it will act as key pair value"""
@login_required
def profile_view(request):
    if request.method == "POST":
        user = request.user
        username = request.POST.get("username")
        email = request.POST.get("email")
        new_password = request.POST.get("new_password")
        confirm_password = request.POST.get("confirm_password")

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
                return redirect('profile')

        user.save()
        messages.success(request, "Profile updated successfully.")
        return redirect('profile')

    # Get user's search history
    search_history = SearchHistory.objects.filter(user=request.user)[:10]  # Get last 10 searches
    
    return render(request, "profile.html", {
        'search_history': search_history
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
    # Record search history if user is authenticated
    if request.user.is_authenticated:
        SearchHistory.objects.create(
            user=request.user,
            search_query=category,
            search_type='category'
        )

    category_cleaned = category.replace('#', '').lower()

    latest_crowd = CrowdData.objects.filter(
        place=OuterRef('pk')
    ).order_by('-timestamp')

    places = Place.objects.filter(
        category__icontains=category_cleaned
    ).annotate(
        latest_crowdlevel=Subquery(latest_crowd.values('crowdlevel')[:1]),
        latest_status=Subquery(latest_crowd.values('status')[:1])
    ).order_by('-latest_crowdlevel')

    places_data = [{
        'id': p.id,
        'name': p.name,
        'latitude': p.latitude,
        'longitude': p.longitude,
        'description': p.description,
        'category': p.category,
        'crowdlevel': p.latest_crowdlevel or 0,
        'status': p.latest_status or 'Unknown',
        'tags': [tag.name for tag in p.tags.all()],
    } for p in places]

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
@api_view(['GET'])
def recommend_places(request, place_name):
    place = get_object_or_404(Place, name__iexact=place_name)
    selected_tags = place.tags.all()
    selected_tag_names = set(selected_tags.values_list('name', flat=True))

    user = request.user if request.user.is_authenticated else None
    user_preference_tags = set()
    if user and hasattr(user, 'userpreference'):
        user_preference_tags = set(user.userpreference.tags.values_list('name', flat=True))

    
    related_places = Place.objects.filter(tags__in=selected_tags).exclude(id=place.id).distinct()

    if related_places.count() < 3:
        related_places = Place.objects.filter(category=place.category).exclude(id=place.id)

    place_scores = []
    for p in related_places:
        other_tag_names = set(p.tags.values_list('name', flat=True))
        shared_tags = selected_tag_names.intersection(other_tag_names)
        weighted_score = sum(TAG_WEIGHTS.get(tag, 1) for tag in shared_tags)
        normalized_score = weighted_score / max(sum(TAG_WEIGHTS.get(tag, 1) for tag in selected_tag_names), 1)
        preference_boost = len(other_tag_names.intersection(user_preference_tags)) * 0.1
        final_score = normalized_score + preference_boost

        place_data = PlaceSerializer(p).data
        place_data['visitor_count'] = random.randint(50, 500)
        place_data['score'] = round(final_score, 2)
        place_scores.append((place_data, final_score))

    sorted_places = sorted(place_scores, key=lambda x: x[1], reverse=True)
    top_places = [data for data, score in sorted_places[:5]]

    return Response({
        "base_place": PlaceSerializer(place).data,
        "recommended": top_places
    })


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

"""List of places"""
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

            # Save the location data
            user_location = UserLocation.objects.create(
                user=request.user,
                latitude=latitude,
                longitude=longitude
            )

            return JsonResponse({"message": "Location saved successfully!"}, status=200)
        except Exception as e:
            return JsonResponse({"error": str(e)}, status=400)
        
@api_view(['GET'])
@login_required  # Ensure the user is authenticated
def get_user_location(request):
    try:
        # Get the most recent location saved by the logged-in user
        user_location = UserLocation.objects.filter(user=request.user).last()

        if user_location:
            return JsonResponse({
                'latitude': user_location.latitude,
                'longitude': user_location.longitude
            })
        else:
            return JsonResponse({"message": "No location found for the user."}, status=404)

    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)
    
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
            'name': place.name,
            'latitude': place.latitude,
            'longitude': place.longitude,
            'description': place.description,
            'popular_for': place.popular_for,
            'category': place.category,
            'crowd_level': crowd_data.crowdlevel if crowd_data else 'N/A',
            'tags': tags,
        })
    
    return JsonResponse({'places': places_data}, safe=False)

@api_view(['GET'])
def places_by_district(request, district_name):
    # Record search history if user is authenticated
    if request.user.is_authenticated:
        SearchHistory.objects.create(
            user=request.user,
            search_query=district_name,
            search_type='district'
        )

    # Fetch places matching the district
    places = Place.objects.filter(district__iexact=district_name)
    result = []

    for place in places:
        latest_crowd = CrowdData.objects.filter(place=place).order_by('-timestamp').first()

        result.append({
            'id': place.id,
            'name': place.name,
            'description': place.description,
            'popular_for': place.popular_for,
            'category': place.category,
            'crowdlevel': latest_crowd.crowdlevel if latest_crowd else None,
            'status': latest_crowd.status if latest_crowd else None,
            'tags': list(place.tags.values_list('name', flat=True)),
            'latitude': place.latitude,
            'longitude': place.longitude
        })

    return JsonResponse({'places': result})


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
            'lng': place.longitude
        })

    return JsonResponse({'places': result})

@login_required
def add_place(request):
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
            tags = request.POST.get('tags', '').split(',')
            crowdlevel = request.POST.get('crowdlevel')
            status = request.POST.get('status')

            # Debug print
            print(f"Received data: {request.POST}")
            print(f"Received files: {request.FILES}")

            # Validate required fields
            if not all([name, description, popular_for, category, district, location, latitude, longitude, image, crowdlevel, status]):
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
                if not status: missing_fields.append('status')
                
                messages.error(request, f'Missing required fields: {", ".join(missing_fields)}')
                return redirect('add_place')

            # Create new place
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
                added_by=request.user
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
                crowdlevel=int(crowdlevel),
                status=status
            )

            messages.success(request, 'Place added successfully!')
            return redirect('place_details', place_id=place.id)
        except Exception as e:
            print(f"Error adding place: {str(e)}")  # Debug print
            messages.error(request, f'Error adding place: {str(e)}')
            return redirect('add_place')
    
    return render(request, 'addplace.html')

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

    return render(request, 'profile.html')

@login_required
def add_place_form(request):
    return render(request, 'addplace.html')

@api_view(['GET'])
@login_required
def get_search_history(request):
    search_history = SearchHistory.objects.filter(user=request.user).order_by('-timestamp')[:10]
    searches = [{
        'search_query': search.search_query,
        'search_type': search.search_type,
        'timestamp': search.timestamp.isoformat()
    } for search in search_history]
    
    return JsonResponse({'searches': searches})