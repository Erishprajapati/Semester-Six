# Quick Reference Guide - Time-Based Crowd Prediction Model

## 🚀 Quick Start

### 1. Generate Training Data
```bash
python manage.py generate_time_based_crowd_data
```

### 2. Train the Model
```bash
python manage.py train_improved_crowd_model --csv-file time_based_crowd_data.csv --force-retrain
```

### 3. Test Predictions
```bash
python demo_time_based_predictions.py
```

## 📊 Time Slot Definitions

| Time Slot | Hours | Description |
|-----------|-------|-------------|
| **Morning** | 5:00 AM - 10:00 AM | Peak for temples, parks, morning routines |
| **Afternoon** | 10:00 AM - 5:00 PM | Peak for markets, tourist attractions, cultural sites |
| **Evening** | 5:00 PM - 10:00 PM | Peak for nightlife, evening walks, entertainment |

## 🏛️ Place Type Patterns

### Religious Places (Temples)
- **Best Time**: Morning (prayers)
- **Avoid**: Afternoon (moderate crowds)
- **Pattern**: High morning → Moderate afternoon → High evening

### Nature/Parks
- **Best Time**: Morning or Evening
- **Avoid**: Afternoon (heat)
- **Pattern**: High morning → Low afternoon → High evening

### Markets (Shopping)
- **Best Time**: Morning (less crowded)
- **Peak Time**: Afternoon/Evening
- **Pattern**: Low morning → High afternoon → Peak evening

### Tourist Attractions
- **Best Time**: Morning (less crowded)
- **Peak Time**: Afternoon
- **Pattern**: Moderate morning → Peak afternoon → Good evening

### Cultural Sites (Museums)
- **Best Time**: Morning (less crowded)
- **Peak Time**: Afternoon
- **Pattern**: Low morning → Peak afternoon → Moderate evening

### Entertainment (Thamel)
- **Best Time**: Morning (dead)
- **Peak Time**: Evening (nightlife)
- **Pattern**: Dead morning → Moderate afternoon → Peak evening

## 🔧 API Usage

### Basic Prediction
```python
from backend.improved_ml_model import ImprovedCrowdPredictionModel

model = ImprovedCrowdPredictionModel()
model.load_model()

crowd_level = model.predict(
    place_id=1,
    category='Temple',
    district='Kathmandu',
    time_slot='morning',  # 'morning', 'afternoon', 'evening'
    day_of_week=6,        # 0=Monday, 6=Sunday
    hour=8,               # 8 for morning, 14 for afternoon, 19 for evening
    month=12,             # 1-12
    season='Winter',      # 'Winter', 'Spring', 'Summer', 'Autumn'
    is_weekend=1,         # 0 or 1
    is_holiday=0,         # 0 or 1
    weather_condition='Sunny'  # 'Sunny', 'Cloudy', 'Rainy', 'Foggy'
)
```

### Time Slot Analysis
```python
# Compare all time slots for a place
time_slots = ['morning', 'afternoon', 'evening']
for slot in time_slots:
    hour = 8 if slot == 'morning' else 14 if slot == 'afternoon' else 19
    crowd = model.predict(
        place_id=1,
        category='Temple',
        district='Kathmandu',
        time_slot=slot,
        hour=hour,
        # ... other parameters
    )
    print(f"{slot}: {crowd:.1f}%")
```

## 📈 Expected Crowd Levels

### Low Crowd (< 30%)
- Markets in morning
- Entertainment areas in morning
- Cultural sites in morning

### Medium Crowd (30-70%)
- Religious places in morning/afternoon
- Tourist attractions in morning
- Cultural sites in evening

### High Crowd (> 70%)
- Nature areas in morning/evening
- Markets in afternoon/evening
- Tourist attractions in afternoon
- Cultural sites in afternoon
- Entertainment areas in evening

## 🎯 Best Practices

### For Tourists
1. **Visit temples early morning** for prayers and less crowds
2. **Go to parks morning or evening** to avoid afternoon heat
3. **Shop in markets morning** for less crowds
4. **Visit tourist sites morning** for better photos
5. **Explore cultural sites afternoon** (indoor activity)
6. **Experience nightlife evening** (Thamel, etc.)

### For Locals
1. **Exercise in parks morning/evening**
2. **Shop in markets morning** for fresh produce
3. **Visit temples during prayer times**
4. **Avoid tourist areas during peak hours**

### For Businesses
1. **Staff markets heavily afternoon/evening**
2. **Open cultural sites all day** (peak afternoon)
3. **Plan entertainment venues for evening focus**
4. **Adjust temple hours for prayer times**

## 🔍 Troubleshooting

### Common Issues
1. **Model not found**: Run training command first
2. **Unknown category**: Use one of the supported categories
3. **Invalid time slot**: Use 'morning', 'afternoon', or 'evening'
4. **Weather not recognized**: Use 'Sunny', 'Cloudy', 'Rainy', or 'Foggy'

### Performance Tips
1. **Load model once** and reuse for multiple predictions
2. **Batch predictions** for multiple places
3. **Cache results** for frequently requested predictions

## 📞 Support

For issues or questions:
1. Check the `TIME_BASED_CROWD_MODEL_SUMMARY.md` for detailed documentation
2. Run `python demo_time_based_predictions.py` to see examples
3. Review the model training logs for performance metrics

## 🎉 Success Indicators

✅ **Model loads successfully** without errors
✅ **Predictions are logical** (morning temples > evening markets)
✅ **Time slots work correctly** (morning < afternoon < evening for markets)
✅ **Weather affects predictions** (rain increases indoor activity)
✅ **Weekend/holiday boosts** are applied correctly

---

**Remember**: This model is trained on realistic Nepali cultural patterns and should provide accurate, culturally-aware crowd predictions! 🇳🇵 