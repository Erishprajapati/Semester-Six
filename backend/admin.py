# Register your models here.
from django.contrib import admin
from .models import *
from django import forms


admin.site.register(Tag)
admin.site.register(CrowdData)
admin.site.register(UserLocation)
admin.site.register(UserPreference)

class PlaceAdminForm(forms.ModelForm):
    class Meta:
        model = Place
        fields = '__all__'
        

    def clean_tags(self):
        tags = self.cleaned_data['tags']
        if tags.count() > 4:
            raise forms.ValidationError("You can select a maximum of 4 tags.")
        return tags

class PlaceAdmin(admin.ModelAdmin):
    form = PlaceAdminForm
    list_display = ('name', 'description', 'popular_for')
    list_filter = ('name', 'popular_for')


admin.site.register(Place, PlaceAdmin)

