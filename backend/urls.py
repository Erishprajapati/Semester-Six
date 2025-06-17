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
    path('profile/',views.profile_view, name = 'profile_view' ),
    # path('placedetails/<int:place_id>/', views.place_details, name='place_details'),
    path('place-details/<int:place_id>/', views.place_details, name='place_details'),
    path('place-details/<int:place_id>/delete/', views.delete_place, name='delete_place'),
    path('place-details/<int:place_id>/save/', views.save_place, name='save_place'),
    # path('saved-places/', views.saved_places, name='saved_places'),
    path('place-list/', views.place_list, name='place_list'),
    # path('places_by_category/<str:category>', views.places_by_category, name='places_by_category'),
    path('add-place/', views.add_place, name='add_place'),
    path('add-place-form/', views.add_place_form, name='add_place_form'),
    path('update-profile/', views.update_profile, name='update_profile'),
    path('places-by-tag/<str:tag_name>/', views.places_by_tag, name='places_by_tag'),
    path('search-history/', views.get_search_history, name='get_search_history'),
    path('predict-crowd/', views.predict_crowd, name='predict_crowd'),

    # path('place/<int:place_id>/', views.place_details, name='place_details'),
    
]
