from django.core.management.base import BaseCommand
from backend.models import Place
import requests
import time

class Command(BaseCommand):
    help = 'Fill missing latitude and longitude for all places using Nominatim geocoding API.'

    def handle(self, *args, **options):
        updated = 0
        skipped = 0
        for place in Place.objects.all():
            if place.latitude and place.longitude:
                skipped += 1
                continue
            # Build query string
            query = f"{place.name}, {place.district or ''}, Nepal"
            url = f"https://nominatim.openstreetmap.org/search"
            params = {
                'q': query,
                'format': 'json',
                'limit': 1
            }
            print(f"Fetching coordinates for: {query}")
            try:
                response = requests.get(url, params=params, headers={'User-Agent': 'PeakTimes/1.0'})
                data = response.json()
                if data:
                    lat = float(data[0]['lat'])
                    lon = float(data[0]['lon'])
                    place.latitude = lat
                    place.longitude = lon
                    place.save()
                    updated += 1
                    print(f"Updated: {place.name} -> ({lat}, {lon})")
                else:
                    print(f"No result for: {query}")
            except Exception as e:
                print(f"Error fetching for {place.name}: {e}")
            time.sleep(1)  # Be polite to Nominatim API
        print(f"Done. Updated: {updated}, Skipped (already had coords): {skipped}") 