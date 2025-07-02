from django.contrib import admin
from django.urls import path
from .views import *
from . import views

urlpatterns = [
    path('', views.register_view, name='register'),
    path('crowd/<str:place_name>', get_crowd_data, name="get_crowd_data"),
    path('recommend/<str:place_name>/', views.recommend_places, name='recommend_places'),
    path('generate-fake/', views.generate_fake_data, name='generate_fake_data'),
    path('places/', views.place_list, name='place_list'),
    path('places-by-district/<str:district_name>/', views.places_by_district, name='places_by_district'),
    path('save-location/', views.save_user_location, name='save_location'),
    path('get-user-location/', views.get_user_location, name='get_user_location'),
    path('search/', views.search_places, name='search_places'),
    path('profile/',views.profile_view, name = 'profile_view' ),
    path('place-details/<int:place_id>/', views.place_details, name='place_details'),
    path('place-details/<int:place_id>/delete/', views.delete_place, name='delete_place'),
    path('place-details/<int:place_id>/save/', views.save_place, name='save_place'),
    path('place-list/', views.place_list, name='place_list'),
    path('add-place/', views.add_place, name='add_place'),
    path('add-place-form/', views.add_place_form, name='add_place_form'),
    path('update-profile/', views.update_profile, name='update_profile'),
    path('places-by-tag/<str:tag_name>/', views.places_by_tag, name='places_by_tag'),
    path('search-history/', views.get_search_history, name='get_search_history'),
    path('api/search-history/', views.get_search_history, name='api_get_search_history'),
    path('predict-crowd/', views.predict_crowd, name='predict_crowd'),
    path('improved-crowd-predictions/', views.improved_crowd_predictions, name='improved_crowd_predictions'),
    path('tourism-crowd-charts/', views.tourism_crowd_data_for_charts, name='tourism_crowd_charts'),
    path('place/<int:place_id>/', views.api_update_place, name='api_update_place'),
    path('balanced-crowd-predictions/', views.balanced_crowd_predictions, name='balanced_crowd_predictions'),
    # Admin approval URLs
    path('pending-places/', views.pending_places, name='pending_places'),
]