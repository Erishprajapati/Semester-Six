from django.core.management.base import BaseCommand
from backend.models import Place, CrowdPattern
import pandas as pd

class Command(BaseCommand):
    help = 'Import crowd pattern data from CSV into the database'

    def handle(self, *args, **options):
        csv_file = 'enhanced_crowd_training_data.csv'
        self.stdout.write(f'Reading {csv_file}...')
        df = pd.read_csv(csv_file)
        count = 0
        for _, row in df.iterrows():
            place_id = int(row['place_id'])
            hour = int(row['hour'])
            crowdlevel = float(row['crowdlevel'])
            place = Place.objects.filter(id=place_id).first()
            if not place:
                self.stdout.write(self.style.WARNING(f'Place not found for id: {place_id}'))
                continue
            obj, created = CrowdPattern.objects.update_or_create(
                place=place, hour=hour,
                defaults={'crowdlevel': crowdlevel}
            )
            count += 1
        self.stdout.write(self.style.SUCCESS(f'Imported/updated {count} crowd pattern records.')) 