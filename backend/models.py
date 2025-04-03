from django.db import models

# Create your models here.
class Place(models.Model):
    name = models.CharField(max_length=255)
    latitude = models.FloatField()
    """float field is used because of ORM can store decimal values as (2.11, 29.01)"""
    longitude = models.FloatField()
    description = models.TextField()
    """Textfield is used because the information can be written more and more"""
    popular_for = models.TextField()

    def __str__(self):
        return f"{self.name} - {self.description} - {self.popular_for}"
    

class CrowdData(models.Model):
    place = models.ForeignKey(Place, on_delete=models.CASCADE)
    timestamp = models.DateTimeField(auto_now_add=True)
    """Crowd data will be updated time to time it uses datetimefield"""
    crowdlevel = models.IntegerField()
    """0 (low) to 100 (high)"""

    
