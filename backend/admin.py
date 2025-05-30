# Register your models here.
from django.contrib import admin
from .models import *
from django import forms
from django.utils.html import format_html
from django.contrib.admin import AdminSite

admin.site.site_header = 'Tourism Management System'
admin.site.site_title = 'Tourism Admin Portal'
admin.site.index_title = 'Welcome to Tourism Management Portal'

class TagAdmin(admin.ModelAdmin):
    list_display = ('name', 'get_place_count')
    search_fields = ('name',)
    
    def get_place_count(self, obj):
        return obj.place_set.count()
    get_place_count.short_description = 'Number of Places'

class UserLocationAdmin(admin.ModelAdmin):
    list_display = ('user', 'latitude', 'longitude', 'created_at')
    list_filter = ('created_at',)
    search_fields = ('user__username',)
    readonly_fields = ('created_at',)

class UserPreferenceAdmin(admin.ModelAdmin):
    list_display = ('user', 'get_tags')
    search_fields = ('user__username', 'tags__name')
    filter_horizontal = ('tags',)
    
    def get_tags(self, obj):
        return ", ".join([tag.name for tag in obj.tags.all()])
    get_tags.short_description = 'Preferred Tags'

class PlaceAdminForm(forms.ModelForm):
    class Meta:
        model = Place
        fields = '__all__'
        widgets = {
            'description': forms.Textarea(attrs={'rows': 4}),
            'popular_for': forms.Textarea(attrs={'rows': 3}),
        }

    def clean_tags(self):
        tags = self.cleaned_data['tags']
        if tags.count() > 4:
            raise forms.ValidationError("You can select a maximum of 4 tags.")
        return tags

class PlaceAdmin(admin.ModelAdmin):
    form = PlaceAdminForm
    list_display = ('name', 'category', 'district', 'location', 'get_tags', 'get_crowd_status', 'added_by')
    list_filter = ('category', 'district', 'tags')
    search_fields = ('name', 'description', 'popular_for', 'location')
    readonly_fields = ('added_by',)
    filter_horizontal = ('tags',)
    fieldsets = (
        ('Basic Information', {
            'fields': ('name', 'description', 'popular_for', 'category', 'district')
        }),
        ('Location Details', {
            'fields': ('location', 'latitude', 'longitude')
        }),
        ('Additional Information', {
            'fields': ('image', 'tags', 'added_by')
        }),
    )
    
    def get_tags(self, obj):
        return ", ".join([tag.name for tag in obj.tags.all()])
    get_tags.short_description = 'Tags'
    
    def get_crowd_status(self, obj):
        latest_crowd = obj.crowddata_set.order_by('-timestamp').first()
        if latest_crowd:
            color = {
                'High': 'red',
                'Medium': 'orange',
                'Low': 'green'
            }.get(latest_crowd.status, 'gray')
            return format_html(
                '<span style="color: {};">{}</span>',
                color,
                f"{latest_crowd.status} ({latest_crowd.crowdlevel}%)"
            )
        return "No Data"
    get_crowd_status.short_description = 'Current Crowd Status'
    
    def save_model(self, request, obj, form, change):
        if not change:  # If creating new object
            obj.added_by = request.user
        super().save_model(request, obj, form, change)

class CrowdDataAdmin(admin.ModelAdmin):
    list_display = ('place', 'crowdlevel', 'status', 'timestamp', 'get_status_color')
    list_filter = ('status', 'timestamp')
    search_fields = ('place__name',)
    list_editable = ('status', 'crowdlevel')
    list_display_links = ('place',)
    readonly_fields = ('timestamp',)
    
    def get_status_color(self, obj):
        color = {
            'High': 'red',
            'Medium': 'orange',
            'Low': 'green'
        }.get(obj.status, 'gray')
        return format_html(
            '<span style="color: {};">{}</span>',
            color,
            obj.status
        )
    get_status_color.short_description = 'Status'

# Register models with their admin classes
admin.site.register(Tag, TagAdmin)
admin.site.register(UserLocation, UserLocationAdmin)
admin.site.register(UserPreference, UserPreferenceAdmin)
admin.site.register(Place, PlaceAdmin)
admin.site.register(CrowdData, CrowdDataAdmin)
