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

# Create your views here.

def home(request):
    return JsonResponse({"message" : "Welcome to System"})
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
        fake_count = random.randint(10, 500)
        status = "High" if fake_count > 300 else "Medium" if fake_count > 100 else "Low"

        CrowdData.objects.create(place=place, crowd_count = fake_count, status = status, timestamp = now())

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

    top_places = [PlaceSerializer(p[0]).data for p in sorted_places[:5]]  # top 5 similar places

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


def show_map(request):
    return render(request, 'Map.html')


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
