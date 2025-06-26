from rest_framework import serializers
from .models import *

class PlaceSerializer(serializers.ModelSerializer):
    tags = serializers.ListField(child=serializers.CharField(), required=False, write_only=True)
    image = serializers.ImageField(required=False, allow_null=True)

    class Meta:
        model = Place
        fields = '__all__'

    def validate_tags(self, tags):
        if len(tags) > 4:
            raise serializers.ValidationError("You can assign at most 4 tags.")
        return tags

    def validate_image(self, value):
        # Allow empty/None values for image
        if value is None or value == '':
            return None
        # If it's a file object, validate it
        if hasattr(value, 'content_type'):
            if not value.content_type.startswith('image/'):
                raise serializers.ValidationError("File must be an image.")
            if value.size > 5 * 1024 * 1024:  # 5MB limit
                raise serializers.ValidationError("Image file size must be less than 5MB.")
        return value

    def create(self, validated_data):
        tags_data = validated_data.pop('tags', [])
        place = super().create(validated_data)
        self._set_tags(place, tags_data)
        return place

    def update(self, instance, validated_data):
        tags_data = validated_data.pop('tags', None)
        # Only remove image if it's explicitly set to None or empty string
        if 'image' in validated_data:
            image_value = validated_data['image']
            if image_value is None or image_value == '':
                validated_data.pop('image')
        place = super().update(instance, validated_data)
        if tags_data is not None:
            self._set_tags(place, tags_data)
        return place

    def _set_tags(self, place, tags_data):
        if not tags_data:
            return
        place.tags.clear()
        for tag_name in tags_data:
            if tag_name and tag_name.strip():
                tag, _ = Tag.objects.get_or_create(name=tag_name.strip())
                place.tags.add(tag)

    def to_representation(self, instance):
        """Custom representation to include tags as a list of names"""
        data = super().to_representation(instance)
        data['tags'] = [tag.name for tag in instance.tags.all()]
        return data

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