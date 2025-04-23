from rest_framework import serializers
from .models import *

class PlaceSerializer(serializers.ModelSerializer):
    class Meta:
        model = Place
        fields = '__all__'

    def validate_tags(self, tags):
        if len(tags) > 4:
            raise serializers.ValidationError("You can assign at most 4 tags.")
        return tags

class CrowdDataSerializer(serializers.ModelSerializer):
    place = PlaceSerializer()
    crowd_status = serializers.SerializerMethodField()

    class Meta:
        model = CrowdData
        fields = '__all__'

    def get_crowd_status(self, obj):
        if obj.crowdlevel > 70:
            return "High"
        elif obj.crowdlevel > 30:
            return "Medium"
        else:
            return "Low"

class TagSerializer(serializers.ModelSerializer):
    class Meta:
        model = Tag
        fields = '__all__'