from django.db import models
from django.contrib.auth.models import User

class Tag(models.Model):
    name = models.CharField(max_length=255)
    
    def __str__(self):
        return self.name

class Place(models.Model):
    name = models.CharField(max_length=255)
    description = models.TextField()
    popular_for = models.TextField()
    category = models.CharField(max_length=50, default='Travel')
    tags = models.ManyToManyField('Tag', blank=True)
    location = models.TextField(default="Unknown")
    district = models.CharField(max_length=100, default="Unknown")  # NEW FIELD

    def __str__(self):
        return f"{self.name}"


class CrowdData(models.Model):
    place = models.ForeignKey(Place, on_delete=models.CASCADE)
    timestamp = models.DateTimeField(auto_now_add=True)
    crowdlevel = models.IntegerField()  # 0 (low) to 100 (high)
    status = models.CharField(
        max_length=255, 
        choices=[
            ('High', 'High'),
            ('Medium', 'Medium'),
            ('Low', 'Low'),
        ]
    )

    def __str__(self):
        return f"{self.place} - {self.timestamp}"

class UserLocation(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    latitude = models.FloatField()
    longitude = models.FloatField()
    created_at = models.DateTimeField(auto_now_add=True)  # Tracks last activity

    def __str__(self):
        return f"{self.user} at {self.latitude}, {self.longitude}"

class UserPreference(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    tags = models.ManyToManyField(Tag)

    def __str__(self):
        return f"{self.user} Preferences"
