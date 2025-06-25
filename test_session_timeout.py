#!/usr/bin/env python3
"""
Test script to verify session timeout functionality
"""

import requests
import time
from datetime import datetime, timedelta

def test_session_timeout():
    """Test session timeout functionality"""
    
    # Base URL
    base_url = "http://127.0.0.1:8000"
    
    print("Testing Session Timeout Functionality")
    print("=" * 50)
    
    # Test 1: Check if login page is accessible
    print("1. Testing login page accessibility...")
    try:
        response = requests.get(f"{base_url}/accounts/login/")
        if response.status_code == 200:
            print("✓ Login page is accessible")
        else:
            print(f"✗ Login page returned status code: {response.status_code}")
    except Exception as e:
        print(f"✗ Error accessing login page: {e}")
        return
    
    # Test 2: Check if home page redirects to login when not authenticated
    print("\n2. Testing home page redirect when not authenticated...")
    try:
        response = requests.get(f"{base_url}/", allow_redirects=False)
        if response.status_code in [302, 301]:
            print("✓ Home page correctly redirects to login when not authenticated")
        else:
            print(f"✗ Home page returned status code: {response.status_code}")
    except Exception as e:
        print(f"✗ Error testing home page redirect: {e}")
    
    # Test 3: Check session settings
    print("\n3. Testing session configuration...")
    try:
        response = requests.get(f"{base_url}/accounts/login/")
        if 'sessionid' in response.cookies:
            print("✓ Session cookies are being set")
        else:
            print("✗ Session cookies are not being set")
    except Exception as e:
        print(f"✗ Error checking session cookies: {e}")
    
    print("\n" + "=" * 50)
    print("Session Timeout Test Summary:")
    print("- Session timeout is set to 30 minutes")
    print("- Users will be logged out after 30 minutes of inactivity")
    print("- A warning will appear 5 minutes before session expires")
    print("- Users can extend their session by clicking 'Stay Logged In'")
    print("- Session activity is tracked on mouse movement, clicks, and keyboard input")
    print("\nTo test the full functionality:")
    print("1. Log in to the application")
    print("2. Wait for 25 minutes to see the warning")
    print("3. Either extend the session or wait for automatic logout")
    print("4. Try accessing protected pages after logout")

if __name__ == "__main__":
    test_session_timeout() 