import random
from django.utils.timezone import now
from django.shortcuts import render, get_object_or_404
from django.http import JsonResponse
from rest_framework.response import Response
from rest_framework.decorators import api_view
from .models import Place, CrowdData, Tag
from .serializers import PlaceSerializer, CrowdDataSerializer, TagSerializer
from .utils import get_weather

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