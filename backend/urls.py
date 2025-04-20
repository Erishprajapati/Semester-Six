from django.contrib import admin
from django.urls import path
from .views import *
from . import views


urlpatterns = [
    path('', home, name='home'),
    path('crowd/<str:place_name>', get_crowd_data, name="get_crowd_data"),
    path('recommend/<str:place_name>/', views.recommend_places, name='recommend_places'),
    path('generate-fake/', views.generate_fake_data, name='generate_fake_data'),
    path('places/', views.place_list, name='place_list'),
    # path('places-by-district/<str:district>', views.places_by_district, name='places_by_district'),

    # path('places/<str:district_name>/', views.places_by_district, name='places_by_district'),  # ✅ added
    path('save-location/', views.save_user_location, name='save_location'),
    path('get-user-location/', views.get_user_location, name='get_user_location'),
    path('search/', views.search_places, name='search_places'),
]
