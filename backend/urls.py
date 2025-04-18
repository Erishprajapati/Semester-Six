from django.contrib import admin
from django.urls import path
from .views import *
from . import views

urlpatterns = [
    path('', home, name='home'),
    path('crowd/<str:place_name>',get_crowd_data, name = "get_crowd_data"),
    path('recommend/<str:place_name>/', views.recommend_places, name='recommend_places'),
    path('generate-fake/', views.generate_fake_data, name='generate_fake_data'),
    path('places/', views.place_list, name='place_list'),
    path('map/', views.show_map, name='show_map'),
    path('save-location/', views.save_user_location, name='save_location'),
    path('get-user-location/', views.get_user_location, name='get_user_location'),
]
