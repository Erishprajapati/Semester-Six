# Register your models here.
from django.contrib import admin
from .models import *

admin.site.register(Place)
admin.site.register(Tag)
admin.site.register(CrowdData)
admin.site.register(UserLocation)
admin.site.register(UserPreference)
