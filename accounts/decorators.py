from django.shortcuts import redirect
from django.contrib.auth import logout
from django.contrib import messages
from django.utils import timezone
from datetime import timedelta
from functools import wraps

def session_timeout_required(view_func):
    """
    Decorator to check if user session has timed out.
    If session has timed out, logout user and redirect to login page.
    """
    @wraps(view_func)
    def _wrapped_view(request, *args, **kwargs):
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
        
        return view_func(request, *args, **kwargs)
    
    return _wrapped_view 