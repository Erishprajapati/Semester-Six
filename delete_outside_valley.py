#!/usr/bin/env python
import os
import django

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'mainfolder.settings')
django.setup()

from backend.models import Place

# Kathmandu Valley bounding box
MIN_LAT, MAX_LAT = 27.55, 27.80
MIN_LON, MAX_LON = 85.20, 85.55
ALLOWED_DISTRICTS = ['Kathmandu', 'Bhaktapur', 'Lalitpur']

def main():
    to_delete = []
    for place in Place.objects.all():
        # Check district
        if place.district not in ALLOWED_DISTRICTS:
            to_delete.append(place)
            continue
        # Check lat/lon
        if not (MIN_LAT <= place.latitude <= MAX_LAT and MIN_LON <= place.longitude <= MAX_LON):
            to_delete.append(place)
    print(f"Found {len(to_delete)} places to delete:")
    for p in to_delete:
        print(f"- {p.name} ({p.district}) [{p.latitude}, {p.longitude}]")
    confirm = input("Delete these places? (y/n): ")
    if confirm.lower() == 'y':
        for p in to_delete:
            p.delete()
        print("Deleted.")
    else:
        print("Aborted.")

if __name__ == '__main__':
    main() 