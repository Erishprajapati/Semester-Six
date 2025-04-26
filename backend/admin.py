# Register your models here.
from django.contrib import admin
from .models import *
from django import forms


admin.site.register(Tag)
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

class CrowdDataAdmin(forms.ModelForm):
    class Meta:
        model = CrowdData
        fields = '__all__'
class PlaceAdmin(admin.ModelAdmin):
    form = PlaceAdminForm
    list_display = ('name', 'description', 'popular_for','location', 'category')
    list_filter = ('name', 'popular_for')


admin.site.register(Place, PlaceAdmin)

class CrowdDataAdmin(admin.ModelAdmin):
    form = CrowdDataAdmin
    list_display = ('place', 'crowdlevel', 'status')
    list_filter = ('status', 'crowdlevel')
    search_fields = ('place',)
    list_editable = ('status',)
    list_display_links = ('place',)
admin.site.register(CrowdData, CrowdDataAdmin)
