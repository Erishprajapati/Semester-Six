from django.core.management.base import BaseCommand
import sys
import os

# Add the backend directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from backend.ml_model import CrowdPredictionModel

class Command(BaseCommand):
    help = 'Train the crowd prediction model'

    def handle(self, *args, **options):
        self.stdout.write('Starting crowd prediction model training...')
        
        # Check if enhanced data exists
        csv_file = 'enhanced_crowd_training_data.csv'
        if not os.path.exists(csv_file):
            self.stdout.write(
                self.style.ERROR(f'Training data file {csv_file} not found!')
            )
            self.stdout.write('Please run: python manage.py export_enhanced_crowd_data first.')
            return
        
        try:
            # Initialize and train model
            model = CrowdPredictionModel()
            
            self.stdout.write('Training model with enhanced data...')
            mse, mae, r2 = model.train(csv_file)
            
            self.stdout.write(
                self.style.SUCCESS(
                    f'Model training completed successfully!\n'
                    f'Mean Squared Error: {mse:.2f}\n'
                    f'Mean Absolute Error: {mae:.2f}\n'
                    f'R² Score: {r2:.2f}'
                )
            )
            
            # Test prediction
            self.stdout.write('\nTesting prediction...')
            from datetime import datetime
            
            test_prediction = model.predict(
                place_id=1,
                category='Temple',
                district='Kathmandu',
                time_slot='morning',
                day_of_week=datetime.now().weekday(),
                month=datetime.now().month,
                season='Spring',
                is_weekend=0,
                is_holiday=0,
                weather_condition='Sunny'
            )
            
            self.stdout.write(f'Test prediction: {test_prediction}% crowd level')
            
        except Exception as e:
            self.stdout.write(
                self.style.ERROR(f'Error training model: {str(e)}')
            ) 