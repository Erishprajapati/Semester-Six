"""
URL configuration for mainfolder project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path,include
from backend.views import *
from django.conf import settings
from django.conf.urls.static import static
from backend import views
from accounts.views import home as accounts_home

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', accounts_home, name='home'),
    path('api/', include('backend.urls')),
    path('accounts/', include('accounts.urls')),
    path('api/crowd/<str:place_name>/', get_crowd_data, name='get_crowd_data'),
    path('places-by-district/<str:district_name>/', places_by_district, name='places_by_district'),
    path('places-by-category/<str:category>', views.places_by_category, name='places-by-category'),
    path('place-details/<int:place_id>/', views.place_details, name='place_details'),
] 

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)