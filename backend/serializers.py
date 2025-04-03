from rest_framework import serializers
from .models import *

class PlaceSerializer(serializers.ModelSerializer):
    class Meta:
        model = Place
        fields = '__all__'

class CrowdDataSerializer(serializers.ModelSerializer):
    place = PlaceSerializer 
    """to show place details in API response"""
    class Meta:
        model = CrowdData
        fields = '__all__'
        