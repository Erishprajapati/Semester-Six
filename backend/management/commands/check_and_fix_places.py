from django.core.management.base import BaseCommand
from backend.models import Place
from django.db.models import Q

class Command(BaseCommand):
    help = 'Check and fix places to ensure only Kathmandu, Lalitpur, and Bhaktapur districts are shown'

    def add_arguments(self, parser):
        parser.add_argument(
            '--fix',
            action='store_true',
            help='Fix places by updating incorrect districts or removing invalid places',
        )
        parser.add_argument(
            '--show-all',
            action='store_true',
            help='Show all places in database regardless of district',
        )

    def handle(self, *args, **options):
        allowed_districts = ['Kathmandu', 'Lalitpur', 'Bhaktapur']
        
        if options['show_all']:
            # Show all places in database
            all_places = Place.objects.all().order_by('district', 'name')
            self.stdout.write(f"\n=== ALL PLACES IN DATABASE ({all_places.count()} total) ===")
            for place in all_places:
                self.stdout.write(f"ID: {place.id}, Name: {place.name}, District: {place.district}, Approved: {place.is_approved}")
            
            # Show district distribution
            district_counts = {}
            for place in all_places:
                district = place.district
                district_counts[district] = district_counts.get(district, 0) + 1
            
            self.stdout.write(f"\n=== DISTRICT DISTRIBUTION ===")
            for district, count in sorted(district_counts.items()):
                status = "✓ ALLOWED" if district in allowed_districts else "✗ NOT ALLOWED"
                self.stdout.write(f"{district}: {count} places - {status}")
        
        # Check for places with invalid districts
        invalid_places = Place.objects.exclude(district__in=allowed_districts)
        self.stdout.write(f"\n=== INVALID PLACES ({invalid_places.count()} found) ===")
        
        if invalid_places.exists():
            for place in invalid_places:
                self.stdout.write(f"ID: {place.id}, Name: {place.name}, District: '{place.district}' (should be one of {allowed_districts})")
            
            if options['fix']:
                self.stdout.write(f"\n=== FIXING INVALID PLACES ===")
                # Try to fix places by checking their location or name
                fixed_count = 0
                deleted_count = 0
                
                for place in invalid_places:
                    # Try to determine correct district from location or name
                    location_lower = place.location.lower() if place.location else ""
                    name_lower = place.name.lower()
                    
                    # Check if location contains district hints
                    if any(district.lower() in location_lower for district in allowed_districts):
                        for district in allowed_districts:
                            if district.lower() in location_lower:
                                place.district = district
                                place.save()
                                self.stdout.write(f"Fixed: {place.name} -> {district} (from location)")
                                fixed_count += 1
                                break
                    elif any(district.lower() in name_lower for district in allowed_districts):
                        for district in allowed_districts:
                            if district.lower() in name_lower:
                                place.district = district
                                place.save()
                                self.stdout.write(f"Fixed: {place.name} -> {district} (from name)")
                                fixed_count += 1
                                break
                    else:
                        # If we can't determine the district, delete the place
                        self.stdout.write(f"Deleting: {place.name} (cannot determine district)")
                        place.delete()
                        deleted_count += 1
                
                self.stdout.write(f"\nFixed: {fixed_count} places")
                self.stdout.write(f"Deleted: {deleted_count} places")
        else:
            self.stdout.write("No invalid places found!")
        
        # Show final valid places
        valid_places = Place.objects.filter(district__in=allowed_districts)
        self.stdout.write(f"\n=== VALID PLACES ({valid_places.count()} total) ===")
        
        for district in allowed_districts:
            district_places = valid_places.filter(district=district)
            self.stdout.write(f"{district}: {district_places.count()} places")
            for place in district_places:
                self.stdout.write(f"  - {place.name} (ID: {place.id}, Approved: {place.is_approved})")
        
        # Check for places without coordinates
        places_without_coords = valid_places.filter(Q(latitude__isnull=True) | Q(longitude__isnull=True))
        if places_without_coords.exists():
            self.stdout.write(f"\n=== PLACES WITHOUT COORDINATES ({places_without_coords.count()} found) ===")
            for place in places_without_coords:
                self.stdout.write(f"ID: {place.id}, Name: {place.name}, District: {place.district}")
        
        # Summary
        self.stdout.write(f"\n=== SUMMARY ===")
        self.stdout.write(f"Total places in database: {Place.objects.count()}")
        self.stdout.write(f"Valid places (3 districts): {valid_places.count()}")
        self.stdout.write(f"Invalid places: {Place.objects.exclude(district__in=allowed_districts).count()}")
        self.stdout.write(f"Approved places: {valid_places.filter(is_approved=True).count()}")
        self.stdout.write(f"Pending approval: {valid_places.filter(is_approved=False).count()}") 