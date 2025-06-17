import csv
from django.core.management.base import BaseCommand
from backend.models import CrowdData

class Command(BaseCommand):
    help = 'Export CrowdData to CSV'

    def handle(self, *args, **kwargs):
        with open('crowd_training_data.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['tag', 'time_slot', 'weekday', 'crowdlevel'])
            for data in CrowdData.objects.all():
                tags = data.place.tags.all()
                if not tags:
                    continue
                tag = tags[0].name.lower()
                time_slot = data.time_slot
                weekday = data.date.weekday()
                crowdlevel = data.crowdlevel
                writer.writerow([tag, time_slot, weekday, crowdlevel])
        self.stdout.write(self.style.SUCCESS('Exported crowd data to crowd_training_data.csv'))
