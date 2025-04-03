import random
from django.utils.timezone import now
from django.shortcuts import render
from django.http import JsonResponse
from .models import *
from .serializers import * 
from rest_framework.response import Response
from rest_framework.decorators import api_view
from django.shortcuts import get_object_or_404
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
