from django.core.management.base import BaseCommand
from backend.models import Place

class Command(BaseCommand):
    help = 'Add famous libraries in Kathmandu, Bhaktapur, and Lalitpur'

    def handle(self, *args, **options):
        libraries = [
            {
                'name': 'Kaiser Library',
                'description': "One of Nepal's oldest and most significant libraries, established in 1907. Houses over 28,000 volumes, including rare books, manuscripts, and historical documents. Also a small museum with antiques and art.",
                'category': 'Library',
                'location': 'Kaiser Mahal, Kantipath',
                'district': 'Kathmandu',
                'latitude': 27.714,  # Approximate
                'longitude': 85.315, # Approximate
                'opening_time': '09:00',
                'closing_time': '18:00',
                'best_time_to_visit': 'Morning to Afternoon',
                'is_approved': True,
            },
            {
                'name': 'Nepal National Library',
                'description': "The official national library of Nepal, with a large collection of books, periodicals, and government publications. Resource for researchers, students, and tourists interested in Nepali history and culture.",
                'category': 'Library',
                'location': 'Sanothimi',
                'district': 'Bhaktapur',
                'latitude': 27.671,  # Approximate
                'longitude': 85.389, # Approximate
                'opening_time': '10:00',
                'closing_time': '17:00',
                'best_time_to_visit': 'Morning to Afternoon',
                'is_approved': True,
            },
            {
                'name': 'Dilliraman Kalyani Regmi Memorial Library',
                'description': "Founded by Dr. Dilliraman Regmi, this library and museum is open to the public and researchers. Features a wide range of books, journals, and archaeological materials.",
                'category': 'Library',
                'location': 'Dilli Bazar',
                'district': 'Kathmandu',
                'latitude': 27.707,  # Approximate
                'longitude': 85.327, # Approximate
                'opening_time': '10:00',
                'closing_time': '17:00',
                'best_time_to_visit': 'Morning to Afternoon',
                'is_approved': True,
            },
            {
                'name': 'Asa Safu Kuthi',
                'description': "The largest library of Nepal Bhasa (Newar) language materials, including books, inscriptions, and chronicles. A unique destination for those interested in Newar culture and language.",
                'category': 'Library',
                'location': 'Manka Dhuku, outside Raktakali Temple',
                'district': 'Kathmandu',
                'latitude': 27.710,  # Approximate
                'longitude': 85.317, # Approximate
                'opening_time': '10:00',
                'closing_time': '17:00',
                'best_time_to_visit': 'Morning to Afternoon',
                'is_approved': True,
            },
        ]
        for lib in libraries:
            obj, created = Place.objects.get_or_create(
                name=lib['name'],
                district=lib['district'],
                defaults=lib
            )
            if created:
                self.stdout.write(self.style.SUCCESS(f"Added: {lib['name']}"))
            else:
                self.stdout.write(self.style.WARNING(f"Already exists: {lib['name']}")) 