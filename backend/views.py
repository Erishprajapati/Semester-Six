import random
import json
from django.utils.timezone import now
from django.shortcuts import render, get_object_or_404
from django.http import JsonResponse
from rest_framework.response import Response
from rest_framework.decorators import api_view
from .models import *
from .serializers import PlaceSerializer, CrowdDataSerializer, TagSerializer
from .utils import get_weather
from django.contrib.auth.decorators import login_required
from django.views.decorators.csrf import csrf_exempt
from django.db.models import Max

# Create your views here.

def home(request):
    return render(request, 'register.html')
    """it will act as key pair value"""

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

@api_view(['GET'])
def recommend_places(request, place_name):
    place = get_object_or_404(Place, name__iexact=place_name)
    selected_tags = place.tags.all()

    # Find all places with similar tags
    related_places = Place.objects.filter(tags__in=selected_tags).exclude(id=place.id).distinct()

    # Calculate similarity score (how many tags in common)
    place_scores = []
    selected_tag_names = set(selected_tags.values_list('name', flat=True))

    for p in related_places:
        other_tag_names = set(p.tags.values_list('name', flat=True))
        score = len(selected_tag_names.intersection(other_tag_names))
        place_scores.append((p, score))

    # Sort based on score (descending)
    sorted_places = sorted(place_scores, key=lambda x: x[1], reverse=True)

    # Generate random number of visitors
    top_places = []
    for p, _ in sorted_places[:5]:  # top 5 similar places
        place_data = PlaceSerializer(p).data
        place_data['visitor_count'] = random.randint(50, 500)  # Random number of visitors
        top_places.append(place_data)

    return Response({
        "base_place": PlaceSerializer(place).data,
        "recommended": top_places
    })
@api_view(['GET'])
def generate_fake_data(request):
    generated_fake_crowd_data()
    return Response({"message": "Fake data generated!"})


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

    # Find places that match the search query (e.g., Kathmandu)
    places = Place.objects.filter(name__icontains=query)
    
    if not places:
        return JsonResponse({'error': 'No places found matching the query.'}, status=404)
    
    places_data = []
    
    # Iterate over the found places and gather the necessary data
    for place in places:
        # Get the latest crowd data for each place
        latest_crowd_data = CrowdData.objects.filter(place=place).aggregate(max_timestamp=Max('timestamp'))
        
        # Get the crowd level using the most recent data
        crowd_data = CrowdData.objects.filter(place=place, timestamp=latest_crowd_data['max_timestamp']).first()
        
        # Prepare the tags for this place
        tags = list(place.tags.values_list('tag', flat=True))
        
        places_data.append({
            'name': place.name,
            # 'latitude': place.latitude,
            # 'longitude': place.longitude,
            'description': place.description,
            'popular_for': place.popular_for,
            'category': place.category,
            'crowd_level': crowd_data.crowdlevel if crowd_data else 'N/A',
            # 'status': crowd_data.status if crowd_data else 'N/A',
            'tags': tags,
        })
    
    # Return the data in JSON format
    return JsonResponse({'places': places_data}, safe=False)

@api_view(['GET'])
def places_by_district(request, district_name):
    # Fetch places matching the district
    places = Place.objects.filter(district__iexact=district_name)
    result = []

    for place in places:
        latest_crowd = CrowdData.objects.filter(place=place).order_by('-timestamp').first()

        result.append({
            'name': place.name,
            'description': place.description,
            'popular_for': place.popular_for,
            'category': place.category,
            'crowdlevel': latest_crowd.crowdlevel if latest_crowd else 'N/A',
            'status': latest_crowd.status if latest_crowd else 'N/A',
            'tags': list(place.tags.values_list('name', flat=True)),
            # 'lat': place.latitude,
            # 'lng': place.longitude
        })

    return JsonResponse({'places': result})