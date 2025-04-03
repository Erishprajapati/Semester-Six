from django.shortcuts import render
from django.http import JsonResponse
# Create your views here.

def home(request):
    return JsonResponse({"message" : "Welcome to System"}) 
    """it will act as key pair value"""
    