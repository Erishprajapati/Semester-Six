from django.urls import path
from . import views
from .views import * 


urlpatterns =  [
    path('register/', views.register_user, name='register'),
    path('login/', views.login_user, name='login'),
    path('logout/', views.logout_user, name='logout'),
    path('dashboard/', views.home, name='dashboard'),
    path('graph/', views.graph, name='bargraph'),
    # path('login/map/', views.show_map, name='show_map'),
]