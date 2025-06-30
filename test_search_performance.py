#!/usr/bin/env python3
"""
Test script to measure search performance improvements
"""

import sys
import os
import time
import requests
sys.path.append(os.path.dirname(__file__))

# Add Django settings
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'mainfolder.settings')
import django
django.setup()

def test_search_performance():
    """Test search performance with different queries"""
    print("🔍 Testing Search Performance Improvements")
    print("=" * 60)
    
    # Test queries
    test_queries = [
        {'type': 'place_search', 'query': 'thamel', 'endpoint': '/api/search/?q=thamel'},
        {'type': 'district_search', 'query': 'kathmandu', 'endpoint': '/places-by-district/kathmandu/'},
        {'type': 'category_search', 'query': 'temple', 'endpoint': '/places-by-category/Temple/'},
        {'type': 'tag_search', 'query': 'religious', 'endpoint': '/places-by-tag/religious/'},
    ]
    
    base_url = 'http://127.0.0.1:8000'
    
    for test in test_queries:
        print(f"\n🔍 Testing {test['type']}: {test['query']}")
        print(f"Endpoint: {test['endpoint']}")
        
        # Test multiple times to get average
        times = []
        for i in range(3):
            try:
                start_time = time.time()
                response = requests.get(f"{base_url}{test['endpoint']}", timeout=30)
                end_time = time.time()
                
                duration = (end_time - start_time) * 1000  # Convert to milliseconds
                times.append(duration)
                
                if response.status_code == 200:
                    data = response.json()
                    if 'places' in data:
                        print(f"  Run {i+1}: {duration:.1f}ms - {len(data['places'])} places found")
                    else:
                        print(f"  Run {i+1}: {duration:.1f}ms - Response received")
                else:
                    print(f"  Run {i+1}: {duration:.1f}ms - Error {response.status_code}")
                    
            except Exception as e:
                print(f"  Run {i+1}: Error - {str(e)}")
        
        if times:
            avg_time = sum(times) / len(times)
            min_time = min(times)
            max_time = max(times)
            print(f"  📊 Average: {avg_time:.1f}ms | Min: {min_time:.1f}ms | Max: {max_time:.1f}ms")
            
            # Performance rating
            if avg_time < 500:
                rating = "🚀 Excellent"
            elif avg_time < 1000:
                rating = "✅ Good"
            elif avg_time < 2000:
                rating = "⚠️ Acceptable"
            else:
                rating = "❌ Slow"
            
            print(f"  {rating} performance")
    
    print("\n🎉 Search performance test completed!")
    print("\n📈 Performance Improvements Applied:")
    print("  ✅ Database indexes added for faster queries")
    print("  ✅ prefetch_related() for efficient tag loading")
    print("  ✅ Bulk crowd data queries instead of individual queries")
    print("  ✅ Cached model predictions")
    print("  ✅ Weather data caching (10 minutes)")
    print("  ✅ Crowd pattern caching (1 hour)")
    print("  ✅ Optimized database queries with select_related()")

if __name__ == "__main__":
    test_search_performance() 