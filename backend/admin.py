# Register your models here.
from django.contrib import admin
from .models import Place, Tag, CrowdData

admin.site.register(Place)
admin.site.register(Tag)
admin.site.register(CrowdData)
