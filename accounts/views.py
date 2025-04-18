from django.shortcuts import render,redirect
from django.contrib.auth.models import User
from django.contrib import messages
from django.contrib.auth import authenticate, login,logout
from django.contrib.auth.decorators import login_required

# Create your views here.

def register(request):
    if request.method == "POST":
        username = request.POST['username']
        email = request.POST['email']
        password = request.POST['password']
        confirm_password = request.POST['confirm_password']

        """if password doesnt match"""
        if password != confirm_password:
            messages.error(request, "Unmatched Credentials")
            return redirect('register')
        
        """username already exists"""
        if User.objects.filter(username = username).exists():
            messages.error(request, "Username already exists")
            return redirect('register')
        
        if User.objects.filter(email = email).exists():
            messages.error(request, "email already exists")
            return redirect('register')
        
        user = User.objects.create_user(username= username, email = email, password = password)
        user.save()
        messages.success(request, "Registration successful.You can login to continue")
        return redirect('login')
    
    return render(request, 'register.html')
        
# def login_user(request):
#     if request.method == 'POST':
#         username = request.POST['username']
#         password = request.POST['password']

#         user = authenticate(request, username=username, password=password)

#         if user is not None:
#             login(request, user)
#             return redirect('dashboard')
#         else:
#             messages.error(request, 'Invalid username or password')
#             return redirect('login')

#     return render(request, 'login.html')

def login_user(request):
    if request.method == 'POST':
        username = request.POST['username']
        password = request.POST['password']

        # DEBUG: print info
        print("Username:", username)
        print("Password:", password)

        try:
            user_check = User.objects.get(username=username)
            print("User found in DB:", user_check)
        except User.DoesNotExist:
            print("User does not exist!")

        user = authenticate(request, username=username, password=password)

        if user is not None:
            login(request, user)
            return redirect('dashboard')  # <- make sure this matches your URL name
        else:
            messages.error(request, 'Invalid username or password')
            return redirect('login')

    return render(request, 'login.html')
# Logout
def logout_user(request):
    logout(request)
    messages.success(request, 'You have been logged out.')
    return redirect('login')


# Dashboard (only accessible if logged in)
@login_required(login_url='login')
def dashboard(request):
    return render(request, 'dashboard.html')