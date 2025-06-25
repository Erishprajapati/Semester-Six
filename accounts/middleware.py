from django.shortcuts import redirect
from django.contrib.auth import logout
from django.contrib import messages
from django.utils import timezone
from datetime import timedelta
from django.conf import settings

class SessionTimeoutMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        # Check if user is authenticated
        if request.user.is_authenticated:
            # Get the last activity time from session
            last_activity = request.session.get('last_activity')
            
            if last_activity:
                # Convert string back to datetime
                last_activity = timezone.datetime.fromisoformat(last_activity.replace('Z', '+00:00'))
                
                # Check if 30 minutes have passed since last activity
                if timezone.now() - last_activity > timedelta(minutes=30):
                    # Session expired, log out user
                    logout(request)
                    messages.warning(request, 'Your session has expired due to inactivity. Please log in again.')
                    return redirect('login')
            
            # Update last activity time
            request.session['last_activity'] = timezone.now().isoformat()
        
        response = self.get_response(request)
        return response 