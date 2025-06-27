from django.core.management.base import BaseCommand
import sys
import os

# Add the backend directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from backend.improved_ml_model import ImprovedCrowdPredictionModel

class Command(BaseCommand):
    help = 'Train the improved crowd prediction model with better data and model selection'

    def add_arguments(self, parser):
        parser.add_argument(
            '--csv-file',
            type=str,
            default='improved_crowd_training_data.csv',
            help='Path to the CSV file for training (default: improved_crowd_training_data.csv)'
        )
        parser.add_argument(
            '--force-retrain',
            action='store_true',
            help='Force retraining even if model exists'
        )

    def handle(self, *args, **options):
        csv_file = options['csv_file']
        force_retrain = options['force_retrain']
        
        self.stdout.write('Starting improved crowd prediction model training...')
        
        # Check if training data exists
        if not os.path.exists(csv_file):
            self.stdout.write(
                self.style.ERROR(f'Training data file {csv_file} not found!')
            )
            self.stdout.write('Please run the improved data generator first:')
            self.stdout.write('python improved_crowd_data_generator.py')
            return
        
        try:
            # Initialize improved model
            model = ImprovedCrowdPredictionModel()
            
            # Check if model already exists and user doesn't want to retrain
            if not force_retrain and model.load_model():
                self.stdout.write(
                    self.style.WARNING('Model already exists. Use --force-retrain to retrain.')
                )
                info = model.get_model_info()
                self.stdout.write(f"Current model: {info['model_name']}")
                self.stdout.write(f"R² Score: {info['r2_score']:.4f}")
                self.stdout.write(f"MAE: {info['mae']:.2f}")
                return
            
            self.stdout.write('Training improved model...')
            mse, mae, r2 = model.train(csv_file)
            
            self.stdout.write(
                self.style.SUCCESS(
                    f'Improved model training completed successfully!\n'
                    f'Model: {model.model_performance["model_name"]}\n'
                    f'Mean Squared Error: {mse:.2f}\n'
                    f'Mean Absolute Error: {mae:.2f}\n'
                    f'R² Score: {r2:.4f}'
                )
            )
            
            # Test predictions
            self.stdout.write('\nTesting predictions...')
            from datetime import datetime
            
            # Test case 1: Temple on weekend morning
            test_prediction1 = model.predict(
                place_id=1,
                category='Temple',
                district='Kathmandu',
                time_slot='morning',
                day_of_week=6,  # Sunday
                month=datetime.now().month,
                season='Spring',
                is_weekend=1,
                is_holiday=0,
                weather_condition='Sunny'
            )
            self.stdout.write(f'Test 1 (Temple, Sunday morning, sunny): {test_prediction1}% crowd level')
            
            # Test case 2: Market on weekday afternoon
            test_prediction2 = model.predict(
                place_id=2,
                category='Market',
                district='Kathmandu',
                time_slot='afternoon',
                day_of_week=2,  # Wednesday
                month=datetime.now().month,
                season='Spring',
                is_weekend=0,
                is_holiday=0,
                weather_condition='Cloudy'
            )
            self.stdout.write(f'Test 2 (Market, Wednesday afternoon, cloudy): {test_prediction2}% crowd level')
            
            # Test case 3: Park on weekend evening
            test_prediction3 = model.predict(
                place_id=3,
                category='Park',
                district='Kathmandu',
                time_slot='evening',
                day_of_week=5,  # Saturday
                month=datetime.now().month,
                season='Spring',
                is_weekend=1,
                is_holiday=0,
                weather_condition='Sunny'
            )
            self.stdout.write(f'Test 3 (Park, Saturday evening, sunny): {test_prediction3}% crowd level')
            
            # Show feature importance if available
            if model.feature_importance is not None:
                self.stdout.write('\nTop 5 Most Important Features:')
                for idx, row in model.feature_importance.head().iterrows():
                    self.stdout.write(f'  {row["feature"]}: {row["importance"]:.4f}')
            
        except Exception as e:
            self.stdout.write(
                self.style.ERROR(f'Error training improved model: {str(e)}')
            )
            import traceback
            self.stdout.write(traceback.format_exc()) 