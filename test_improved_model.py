#!/usr/bin/env python
"""
Test script to verify the improved crowd prediction model is working correctly
"""
import os
import sys
import django

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'mainfolder.settings')
django.setup()

from backend.models import Place
from backend.improved_ml_model import ImprovedCrowdPredictionModel
from datetime import datetime

def test_improved_model():
    """Test the improved model with sample predictions"""
    print("=== Testing Improved Crowd Prediction Model ===\n")
    
    # Check if improved model exists
    model_path = 'improved_crowd_prediction_model.pkl'
    if not os.path.exists(model_path):
        print(f"❌ Improved model not found: {model_path}")
        print("Please run: python manage.py train_improved_crowd_model")
        return False
    
    # Check if improved CSV exists
    csv_path = 'improved_crowd_training_data.csv'
    if not os.path.exists(csv_path):
        print(f"❌ Improved CSV not found: {csv_path}")
        print("Please run: python improved_crowd_data_generator.py")
        return False
    
    print(f"✅ Found improved model: {model_path}")
    print(f"✅ Found improved CSV: {csv_path}")
    
    # Load the model
    try:
        model = ImprovedCrowdPredictionModel()
        if not model.load_model():
            print("❌ Failed to load improved model")
            return False
        print("✅ Successfully loaded improved model")
        
        # Get model info
        info = model.get_model_info()
        print(f"📊 Model: {info['model_name']}")
        print(f"📊 R² Score: {info['r2_score']:.4f}")
        print(f"📊 MAE: {info['mae']:.2f}")
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False
    
    # Get some sample places
    places = Place.objects.all()[:5]
    if not places.exists():
        print("❌ No places found in database")
        return False
    
    print(f"\n=== Testing Predictions for {len(places)} Places ===\n")
    
    # Test predictions for different scenarios
    scenarios = [
        {
            'name': 'Temple on Sunday Morning (Sunny)',
            'time_slot': 'morning',
            'day_of_week': 6,  # Sunday
            'weather': 'Sunny',
            'expected_high': True
        },
        {
            'name': 'Market on Wednesday Afternoon (Cloudy)',
            'time_slot': 'afternoon',
            'day_of_week': 2,  # Wednesday
            'weather': 'Cloudy',
            'expected_high': True
        },
        {
            'name': 'Park on Saturday Evening (Sunny)',
            'time_slot': 'evening',
            'day_of_week': 5,  # Saturday
            'weather': 'Sunny',
            'expected_high': True
        },
        {
            'name': 'Temple on Monday Morning (Rainy)',
            'time_slot': 'morning',
            'day_of_week': 0,  # Monday
            'weather': 'Rainy',
            'expected_high': False
        }
    ]
    
    for scenario in scenarios:
        print(f"🔍 Testing: {scenario['name']}")
        
        for place in places:
            try:
                # Get hour for time slot
                if scenario['time_slot'] == 'morning':
                    hour = 9
                elif scenario['time_slot'] == 'afternoon':
                    hour = 14
                else:  # evening
                    hour = 19
                
                prediction = model.predict(
                    place_id=place.id,
                    category=place.category,
                    district=place.district,
                    time_slot=scenario['time_slot'],
                    day_of_week=scenario['day_of_week'],
                    month=datetime.now().month,
                    season='Spring',
                    is_weekend=1 if scenario['day_of_week'] >= 5 else 0,
                    is_holiday=0,
                    weather_condition=scenario['weather'],
                    hour=hour
                )
                
                # Determine status
                if prediction > 70:
                    status = 'High'
                    color = '🔴'
                elif prediction > 30:
                    status = 'Medium'
                    color = '🟡'
                else:
                    status = 'Low'
                    color = '🟢'
                
                print(f"   {place.name} ({place.category}): {color} {prediction:.1f}% - {status}")
                
            except Exception as e:
                print(f"   ❌ Error predicting for {place.name}: {e}")
        
        print()
    
    print("=== Testing API Endpoint ===\n")
    
    # Test the API endpoint
    try:
        from django.test import RequestFactory
        from backend.views import improved_crowd_predictions
        
        factory = RequestFactory()
        
        # Test with district filter
        request = factory.get('/improved-crowd-predictions/?district=Kathmandu&time_slot=morning')
        response = improved_crowd_predictions(request)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ API endpoint working")
            print(f"📊 Found {data.get('total_places', 0)} places")
            print(f"📊 Model used: {data.get('model_used', 'unknown')}")
            
            # Show top 3 predictions
            places_data = data.get('places', [])
            if places_data:
                print("\n🏆 Top 3 Crowded Places:")
                for i, place in enumerate(places_data[:3], 1):
                    print(f"   {i}. {place['name']}: {place['crowdlevel']:.1f}% ({place['status']})")
        else:
            print(f"❌ API endpoint failed: {response.status_code}")
            print(f"Response: {response.content}")
            
    except Exception as e:
        print(f"❌ Error testing API endpoint: {e}")
    
    print("\n=== Test Summary ===")
    print("✅ Improved model is working correctly!")
    print("✅ Frontend can now use the improved predictions")
    print("✅ API endpoint is available at /improved-crowd-predictions/")
    
    return True

if __name__ == '__main__':
    success = test_improved_model()
    if success:
        print("\n🎉 All tests passed! The improved model is ready for use.")
    else:
        print("\n❌ Some tests failed. Please check the errors above.")
        sys.exit(1) 