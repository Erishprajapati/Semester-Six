from django.core.management.base import BaseCommand
import sys
import os
import pandas as pd

# Add the backend directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from backend.improved_ml_model import ImprovedCrowdPredictionModel

class Command(BaseCommand):
    help = 'Train the enhanced crowd prediction model with advanced data cleaning, feature engineering, and SHAP explainability'

    def add_arguments(self, parser):
        parser.add_argument(
            '--csv-file',
            type=str,
            default='nepal_tourism_crowd_data.csv',
            help='Path to the CSV file for training (default: nepal_tourism_crowd_data.csv)'
        )
        parser.add_argument(
            '--force-retrain',
            action='store_true',
            help='Force retraining even if model already exists'
        )
        parser.add_argument(
            '--model-output',
            type=str,
            default='crowd_prediction_model.joblib',
            help='Path to save the trained model (default: crowd_prediction_model.joblib)'
        )

    def handle(self, *args, **options):
        csv_file = options['csv_file']
        force_retrain = options['force_retrain']
        model_output = options['model_output']
        
        # Check if CSV file exists
        if not os.path.exists(csv_file):
            self.stdout.write(
                self.style.ERROR(f'❌ CSV file "{csv_file}" not found!')
            )
            self.stdout.write(
                self.style.WARNING(f'Available CSV files:')
            )
            for file in os.listdir('.'):
                if file.endswith('.csv'):
                    self.stdout.write(f'  • {file}')
            return
        
        # Check if model already exists
        if os.path.exists(model_output) and not force_retrain:
            self.stdout.write(
                self.style.WARNING(f'⚠️ Model already exists at "{model_output}"')
            )
            self.stdout.write(
                self.style.WARNING(f'Use --force-retrain to retrain the model')
            )
            return
        
        self.stdout.write(
            self.style.SUCCESS('🚀 Starting enhanced crowd prediction model training...')
        )
        self.stdout.write(
            self.style.SUCCESS('🧠 Training enhanced model with advanced features...')
        )
        
        try:
            # Initialize the enhanced model
            model = ImprovedCrowdPredictionModel(model_output)
            
            # Train the model
            success = model.train_model(csv_file)
            
            if success:
                # Get model info
                info = model.get_model_info()
                
                self.stdout.write(
                    self.style.SUCCESS('🎉 Enhanced model training completed successfully!')
                )
                self.stdout.write(
                    self.style.SUCCESS(f'🤖 Model: {info.get("model_name", "Unknown")}')
                )
                self.stdout.write(
                    self.style.SUCCESS(f'📊 Mean Squared Error: {info.get("mse", 0):.2f}')
                )
                self.stdout.write(
                    self.style.SUCCESS(f'📏 Mean Absolute Error: {info.get("mae", 0):.2f}')
                )
                self.stdout.write(
                    self.style.SUCCESS(f'🎯 R² Score: {info.get("r2_score", 0):.4f}')
                )
                self.stdout.write(
                    self.style.SUCCESS(f'📈 Data Source: {info.get("data_source", "Unknown")}')
                )
                self.stdout.write(
                    self.style.SUCCESS(f'📊 Total Samples: {info.get("total_samples", 0):,}')
                )
                self.stdout.write(
                    self.style.SUCCESS(f'🔧 Total Features: {info.get("total_features", 0)}')
                )
                
                # Test predictions
                self.stdout.write(
                    self.style.SUCCESS('\n🧪 Testing predictions...')
                )
                
                test_cases = [
                    {
                        'name': 'Temple, Sunday morning, sunny',
                        'params': {
                            'place_id': 1,
                            'category': 'Religious',
                            'district': 'Kathmandu',
                            'time_slot': 'morning',
                            'day_of_week': 6,
                            'month': 12,
                            'season': 'Winter',
                            'is_weekend': 1,
                            'is_holiday': 0,
                            'weather_condition': 'Sunny'
                        }
                    },
                    {
                        'name': 'Market, Wednesday afternoon, cloudy',
                        'params': {
                            'place_id': 2,
                            'category': 'Market',
                            'district': 'Kathmandu',
                            'time_slot': 'afternoon',
                            'day_of_week': 2,
                            'month': 6,
                            'season': 'Summer',
                            'is_weekend': 0,
                            'is_holiday': 0,
                            'weather_condition': 'Cloudy'
                        }
                    },
                    {
                        'name': 'Park, Saturday evening, sunny',
                        'params': {
                            'place_id': 3,
                            'category': 'Park',
                            'district': 'Lalitpur',
                            'time_slot': 'evening',
                            'day_of_week': 5,
                            'month': 4,
                            'season': 'Spring',
                            'is_weekend': 1,
                            'is_holiday': 0,
                            'weather_condition': 'Sunny'
                        }
                    }
                ]
                
                for test_case in test_cases:
                    try:
                        prediction = model.predict(**test_case['params'])
                        self.stdout.write(
                            self.style.SUCCESS(f'🕍 {test_case["name"]}: {prediction:.1f}% crowd level')
                        )
                    except Exception as e:
                        self.stdout.write(
                            self.style.ERROR(f'❌ {test_case["name"]}: Error - {e}')
                        )
                
                # Show feature importance if available
                if hasattr(model.model, 'feature_importances_'):
                    self.stdout.write(
                        self.style.SUCCESS('\n🏆 Top 5 Most Important Features:')
                    )
                    feature_importance = model.model.feature_importances_
                    feature_names = model.feature_names
                    
                    if feature_names and len(feature_names) == len(feature_importance):
                        feature_importance_df = pd.DataFrame({
                            'feature': feature_names,
                            'importance': feature_importance
                        }).sort_values('importance', ascending=False)
                        
                        for i, row in feature_importance_df.head(5).iterrows():
                            self.stdout.write(
                                self.style.SUCCESS(f'  📈 {row["feature"]}: {row["importance"]:.4f}')
                            )
                
                if not info.get('shap_available', False):
                    self.stdout.write(
                        self.style.WARNING('\n⚠️ SHAP not available. Install with: pip install shap')
                    )
                
                self.stdout.write(
                    self.style.SUCCESS('\n📖 Model Usage:')
                )
                self.stdout.write(
                    self.style.SUCCESS('  • The model is automatically used by the API endpoints')
                )
                self.stdout.write(
                    self.style.SUCCESS('  • Use --force-retrain to update the model with new data')
                )
                self.stdout.write(
                    self.style.SUCCESS('  • Check model performance with: python manage.py train_improved_crowd_model')
                )
                
            else:
                self.stdout.write(
                    self.style.ERROR('❌ Model training failed!')
                )
                
        except Exception as e:
            self.stdout.write(
                self.style.ERROR(f'❌ Error during training: {str(e)}')
            )
            import traceback
            self.stdout.write(
                self.style.ERROR(traceback.format_exc())
            ) 
            self.stdout.write(traceback.format_exc()) 