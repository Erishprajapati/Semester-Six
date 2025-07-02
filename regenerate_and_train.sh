#!/bin/bash

set -e

# Step 1: Generate enhanced time-based crowd data
echo "[1/3] Generating enhanced_time_based_crowd_data.csv..."
python manage.py generate_time_based_crowd_data

echo "[2/3] Training improved crowd prediction model with new data..."
python manage.py train_improved_crowd_model --csv-file enhanced_time_based_crowd_data.csv

echo "[3/3] All done!"
echo "If your server is running, restart it to use the new model:"
echo "    python manage.py runserver" 