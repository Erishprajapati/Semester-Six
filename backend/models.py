from django.db import models
from django.contrib.auth.models import User
from django.core.exceptions import ValidationError
from django.utils import timezone

class Tag(models.Model):
    name = models.CharField(max_length=255, db_index=True)
    
    def __str__(self):
        return self.name

class Place(models.Model):
    name = models.CharField(max_length=255, db_index=True)
    description = models.TextField()
    popular_for = models.TextField()
    category = models.CharField(max_length=50, default='Travel', db_index=True)
    tags = models.ManyToManyField('Tag', blank=True)
    location = models.TextField(default="Unknown")
    district = models.CharField(max_length=100, default="Unknown", db_index=True)
    latitude = models.FloatField(null=True, blank=True)
    longitude = models.FloatField(null=True, blank=True)
    image = models.ImageField(upload_to='place_images/', null=True, blank=True)
    added_by = models.ForeignKey(User, on_delete=models.SET_NULL, null=True)
    is_approved = models.BooleanField(default=False, db_index=True)
    
    # Entry fee fields
    tourist_fee_npr = models.DecimalField(max_digits=8, decimal_places=2, null=True, blank=True, help_text="Entry fee for foreign tourists in NPR")
    tourist_fee_usd = models.DecimalField(max_digits=6, decimal_places=2, null=True, blank=True, help_text="Entry fee for foreign tourists in USD")
    local_fee_npr = models.DecimalField(max_digits=8, decimal_places=2, null=True, blank=True, help_text="Entry fee for locals in NPR")
    saarc_fee_npr = models.DecimalField(max_digits=8, decimal_places=2, null=True, blank=True, help_text="Entry fee for SAARC nationals in NPR")
    fee_description = models.TextField(blank=True, help_text="Additional information about fees, discounts, or special conditions")
    has_entry_fee = models.BooleanField(default=False, help_text="Whether this place has an entry fee")

    opening_time = models.TimeField(null=True, blank=True, help_text="Opening time (e.g., 09:00)")
    closing_time = models.TimeField(null=True, blank=True, help_text="Closing time (e.g., 17:00)")
    best_time_to_visit = models.CharField(max_length=100, blank=True, help_text="Best time to visit (e.g., Morning, Evening, Spring)")
    closed_dates = models.CharField(max_length=255, blank=True, help_text="Comma-separated dates or festival names when closed (e.g., 2025-10-24, Holi)")

    class Meta:
        unique_together = ('name', 'district')
        indexes = [
            models.Index(fields=['name', 'district']),
            models.Index(fields=['category', 'district']),
            models.Index(fields=['is_approved', 'district']),
        ]

    def __str__(self):
        return f"{self.name}"

    def clean(self):
        super().clean()
        if self.pk and self.tags.count() > 4:
            raise ValidationError("A place can have at most 4 tags.")


class CrowdData(models.Model):
    TIME_SLOT_CHOICES = [
        ('morning', 'Morning'),
        ('afternoon', 'Afternoon'),
        ('evening', 'Evening'),
    ]
    
    place = models.ForeignKey(Place, on_delete=models.CASCADE, db_index=True)
    timestamp = models.DateTimeField(auto_now_add=True, db_index=True)
    crowdlevel = models.IntegerField()  # 0 (low) to 100 (high)
    status = models.CharField(
        max_length=255, 
        choices=[
            ('High', 'High'),
            ('Medium', 'Medium'),
            ('Low', 'Low'),
        ]
    )
    date = models.DateField(default=timezone.now)
    time_slot = models.CharField(
        max_length=10,
        choices=TIME_SLOT_CHOICES,
        default='morning'
    )

    class Meta:
        indexes = [
            models.Index(fields=['place', 'timestamp']),
            models.Index(fields=['timestamp']),
        ]

    def __str__(self):
        return f"{self.place} - {self.date} {self.time_slot} - {self.status}"

class UserLocation(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, db_index=True)
    latitude = models.FloatField()
    longitude = models.FloatField()
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.user} at {self.latitude}, {self.longitude}"

class UserPreference(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, db_index=True)
    tags = models.ManyToManyField(Tag)

    def __str__(self):
        return f"{self.user} Preferences"

class SearchHistory(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, db_index=True)
    search_query = models.CharField(max_length=255, db_index=True)
    search_type = models.CharField(max_length=50, db_index=True)
    timestamp = models.DateTimeField(auto_now_add=True, db_index=True)
    
    class Meta:
        ordering = ['-timestamp']
        verbose_name_plural = 'Search Histories'
        indexes = [
            models.Index(fields=['user', 'timestamp']),
            models.Index(fields=['search_type', 'timestamp']),
        ]
    
    def __str__(self):
        return f"{self.user.username} searched for {self.search_query} at {self.timestamp}"

class CrowdPattern(models.Model):
    place = models.ForeignKey(Place, on_delete=models.CASCADE, db_index=True)
    hour = models.IntegerField()
    crowdlevel = models.FloatField()

    class Meta:
        unique_together = ('place', 'hour')
        indexes = [
            models.Index(fields=['place', 'hour']),
        ]

    def __str__(self):
        return f"{self.place.name} - {self.hour}:00 ({self.crowdlevel})"

class Review(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    place = models.ForeignKey('Place', on_delete=models.CASCADE, related_name='reviews')
    rating = models.IntegerField(choices=[(i, i) for i in range(1, 6)])
    comment = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Review by {self.user.username} for {self.place.name} ({self.rating}★)"
    
    