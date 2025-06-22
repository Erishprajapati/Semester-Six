#!/usr/bin/env python3
import requests
import json

def test_places_by_district():
    """Test the places-by-district endpoint"""
    try:
        # Test with Kathmandu
        response = requests.get('http://localhost:8000/places-by-district/Kathmandu/')
        print(f"Status Code: {response.status_code}")
        print(f"Response Headers: {dict(response.headers)}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"Response Data: {json.dumps(data, indent=2)}")
        else:
            print(f"Error Response: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("Connection error - make sure the Django server is running on localhost:8000")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_places_by_district() 