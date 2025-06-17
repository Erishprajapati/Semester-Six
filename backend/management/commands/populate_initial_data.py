from django.core.management.base import BaseCommand
from django.contrib.auth.models import User
from backend.models import Tag, Place, CrowdData
from django.utils import timezone
import random

class Command(BaseCommand):
    help = 'Populate initial data for testing'

    def handle(self, *args, **options):
        self.stdout.write('Creating initial data...')
        
        # Create tags
        tags_data = [
            'Temple', 'Museum', 'Park', 'Restaurant', 'Shopping', 
            'Historical', 'Nature', 'Adventure', 'Cultural', 'Religious'
        ]
        
        tags = []
        for tag_name in tags_data:
            tag, created = Tag.objects.get_or_create(name=tag_name)
            tags.append(tag)
            if created:
                self.stdout.write(f'Created tag: {tag_name}')
        
        # Create some sample places
        places_data = [
            {
                'name': 'Pashupatinath Temple',
                'description': 'One of the most sacred Hindu temples dedicated to Lord Shiva',
                'popular_for': 'Religious ceremonies, cultural heritage',
                'category': 'Religious',
                'location': 'Pashupati, Kathmandu',
                'district': 'Kathmandu',
                'latitude': 27.7101,
                'longitude': 85.3483,
                'tags': ['Temple', 'Religious', 'Cultural', 'Historical']
            },
            {
                'name': 'Swayambhunath Stupa',
                'description': 'Ancient religious complex atop a hill in the Kathmandu Valley',
                'popular_for': 'Buddhist pilgrimage, panoramic views',
                'category': 'Religious',
                'location': 'Swayambhu, Kathmandu',
                'district': 'Kathmandu',
                'latitude': 27.7149,
                'longitude': 85.2906,
                'tags': ['Temple', 'Religious', 'Cultural', 'Historical']
            },
            {
                'name': 'Boudhanath Stupa',
                'description': 'One of the largest stupas in Nepal and a UNESCO World Heritage site',
                'popular_for': 'Buddhist pilgrimage, meditation',
                'category': 'Religious',
                'location': 'Boudha, Kathmandu',
                'district': 'Kathmandu',
                'latitude': 27.7218,
                'longitude': 85.3618,
                'tags': ['Temple', 'Religious', 'Cultural']
            },
            {
                'name': 'Kathmandu Durbar Square',
                'description': 'Historic palace complex and UNESCO World Heritage site',
                'popular_for': 'Historical architecture, cultural heritage',
                'category': 'Historical',
                'location': 'Hanuman Dhoka, Kathmandu',
                'district': 'Kathmandu',
                'latitude': 27.7045,
                'longitude': 85.3072,
                'tags': ['Historical', 'Cultural', 'Museum']
            },
            {
                'name': 'Garden of Dreams',
                'description': 'Beautiful neo-classical garden in the heart of Kathmandu',
                'popular_for': 'Relaxation, photography, peaceful environment',
                'category': 'Park',
                'location': 'Kaiser Mahal, Kathmandu',
                'district': 'Kathmandu',
                'latitude': 27.7172,
                'longitude': 85.3240,
                'tags': ['Park', 'Nature', 'Cultural']
            }
        ]
        
        # Get or create admin user
        admin_user, created = User.objects.get_or_create(
            username='admin',
            defaults={
                'email': 'admin@example.com',
                'is_staff': True,
                'is_superuser': True
            }
        )
        
        places = []
        for place_data in places_data:
            place, created = Place.objects.get_or_create(
                name=place_data['name'],
                district=place_data['district'],
                defaults={
                    'description': place_data['description'],
                    'popular_for': place_data['popular_for'],
                    'category': place_data['category'],
                    'location': place_data['location'],
                    'latitude': place_data['latitude'],
                    'longitude': place_data['longitude'],
                    'added_by': admin_user,
                    'is_approved': True
                }
            )
            
            if created:
                # Add tags to place
                for tag_name in place_data['tags']:
                    tag = Tag.objects.get(name=tag_name)
                    place.tags.add(tag)
                
                places.append(place)
                self.stdout.write(f'Created place: {place.name}')
        
        # Create some crowd data
        today = timezone.now().date()
        time_slots = ['morning', 'afternoon', 'evening']
        
        for place in places:
            for time_slot in time_slots:
                crowd_level = random.randint(20, 80)
                status = 'High' if crowd_level > 60 else 'Medium' if crowd_level > 30 else 'Low'
                
                crowd_data, created = CrowdData.objects.get_or_create(
                    place=place,
                    date=today,
                    time_slot=time_slot,
                    defaults={
                        'crowdlevel': crowd_level,
                        'status': status
                    }
                )
                
                if created:
                    self.stdout.write(f'Created crowd data for {place.name} - {time_slot}')
        
        self.stdout.write(
            self.style.SUCCESS('Successfully created initial data!')
        ) 