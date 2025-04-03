from django.contrib import admin
from django.urls import path
from .views import *

urlpatterns = [
    path('', home, name='home'),
    # path('crowd/<str:place_name>',get_crowd_data, name = "get_crowd_data")
]
