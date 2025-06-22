#!/usr/bin/env python3
import requests
import json

def test_recommendations_api():
    """Test the recommendations API to see if it includes image field"""
    test_places = [
        "Swayambhunath Stupa",
        "Pashupatinath Temple", 
        "Thamel",
        "Garden of Dreams",
        "Ichangu Narayan Temple"
    ]
    
    for place_name in test_places:
        try:
            print(f"\n{'='*50}")
            print(f"Testing recommendations for: {place_name}")
            print(f"{'='*50}")
            
            # Test with a place that likely has recommendations
            response = requests.get(f'http://localhost:8000/api/recommend/{place_name}')
            print(f"Status Code: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                
                # Check if recommendations have image field
                if data.get('recommendations'):
                    print(f"Found {len(data['recommendations'])} recommendations!")
                    for i, rec in enumerate(data['recommendations']):
                        print(f"\nRecommendation {i+1}:")
                        print(f"  Name: {rec.get('name')}")
                        print(f"  Image: {rec.get('image')}")
                        print(f"  Has Image: {'Yes' if rec.get('image') else 'No'}")
                        print(f"  Category: {rec.get('category')}")
                        print(f"  Tags: {rec.get('tags')}")
                    return  # Found recommendations, stop testing
                else:
                    print("No recommendations found for this place.")
            else:
                print(f"Error Response: {response.text}")
                
        except requests.exceptions.ConnectionError:
            print("Connection error - make sure the Django server is running on localhost:8000")
            return
        except Exception as e:
            print(f"Error: {e}")
    
    print("\nNo places with recommendations found. This might indicate an issue with the recommendation algorithm.")

if __name__ == "__main__":
    test_recommendations_api() 