from django.db import models
from django.contrib.auth.models import User
from django.core.exceptions import ValidationError

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
    district = models.CharField(max_length=100, default="Unknown")
    latitude = models.FloatField(null=True, blank=True)
    longitude = models.FloatField(null=True, blank=True)
    image = models.ImageField(upload_to='place_images/', null=True, blank=True)  # ✅ New field
    added_by = models.ForeignKey(User, on_delete=models.SET_NULL, null=True)
    is_approved = models.BooleanField(default=False)

    class Meta:
        unique_together = ('name', 'district')

    def __str__(self):
        return f"{self.name}"

    def clean(self):
        super().clean()  # Good practice to call parent clean
        if self.pk and self.tags.count() > 4:
            raise ValidationError("A place can have at most 4 tags.")


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

class SearchHistory(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    search_query = models.CharField(max_length=255)
    search_type = models.CharField(max_length=50)  # 'place', 'district', or 'category'
    timestamp = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['-timestamp']
        verbose_name_plural = 'Search Histories'
    
    def __str__(self):
        return f"{self.user.username} searched for {self.search_query} at {self.timestamp}"
    
    
