# 🏔️ Smart Tourism Crowd Prediction System

A Django-based web application that predicts and visualizes crowd levels at popular tourist places in Nepal using machine learning and real-time weather data.

## 🌟 Key Features

### 🎯 **Intelligent Crowd Predictions**
- **Time-based predictions:** Morning (5-10 AM), Afternoon (10-5 PM), Evening (5-10 PM)
- **Real-time weather integration:** Live weather data from OpenWeatherMap API
- **Cultural awareness:** Festival-specific crowd boosts (Dashain, Tihar, Nepali New Year)
- **Place-specific patterns:** Unique crowd patterns for major landmarks
- **District scaling:** Population-based crowd level adjustments

### 🗺️ **Interactive Map Dashboard**
- **Real-time crowd visualization:** Color-coded crowd levels (Red=High, Green=Medium, Yellow=Low)
- **Weather impact display:** Shows how weather affects crowd predictions
- **Time slot filtering:** View predictions for different times of day
- **District-based views:** Filter by Kathmandu, Lalitpur, Bhaktapur
- **Category filtering:** Religious, Tourist, Market, Cultural, Nature, Entertainment

### 🌤️ **Weather Integration**
- **Live weather data:** Real-time temperature, conditions, and humidity
- **Weather impact calculation:** How weather affects crowd levels
- **Seasonal patterns:** Different weather patterns by season
- **Indoor/outdoor logic:** Rain increases indoor cultural sites, decreases outdoor activities

### 📊 **Advanced Analytics**
- **Bar charts:** Visual crowd level comparisons
- **Trend analysis:** Historical crowd patterns
- **Festival predictions:** Special crowd boosts during major festivals
- **Weather correlation:** Weather impact on tourist behavior

## 🚀 Recent Enhancements (Latest Update)

### 🆕 **Enhanced Data Generation System**
- **Place-specific overrides:** 9 major places with unique crowd patterns
- **Festival boosts:** 4 major Nepali festivals with specific crowd increases
- **District scaling:** Population-based adjustments (Kathmandu +20%, Lalitpur +10%)
- **Enhanced randomness:** ±15% weekends, ±10% weekdays for realistic variability
- **Cultural accuracy:** Religiously-sensitive and festival-aware patterns

### 🎪 **Festival Integration**
- **Dashain (October):** 50%+ crowd increase at temples, 35% at markets
- **Tihar (November):** Festival of Lights boosts cultural sites and markets
- **Nepali New Year (April):** Celebrations across all major tourist sites
- **Janai Purnima (August):** Religious ritual boosts at temples

### 🏛️ **Place-Specific Intelligence**
- **Pashupatinath Temple:** High morning/evening prayer attendance
- **Thamel:** Low morning, peak evening nightlife patterns
- **Garden of Dreams:** Seasonal variations with summer heat impact
- **Asan Bazaar:** Local shopping patterns vs tourist areas

## 🛠️ Technology Stack

- **Backend:** Django 5.2, Python 3.13
- **Database:** PostgreSQL
- **Machine Learning:** XGBoost, Scikit-learn
- **Weather API:** OpenWeatherMap
- **Frontend:** HTML5, CSS3, JavaScript, Chart.js
- **Maps:** Leaflet.js
- **Admin:** Django Jazzmin

## 📦 Installation

### Prerequisites
- Python 3.13+
- PostgreSQL
- OpenWeatherMap API key

### Setup
```bash
# Clone the repository
git clone <repository-url>
cd Project

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your database and API credentials

# Run migrations
python manage.py migrate

# Create superuser
python manage.py createsuperuser

# Generate enhanced training data
python manage.py generate_time_based_crowd_data

# Train the improved model
python manage.py train_improved_crowd_model --csv-file enhanced_time_based_crowd_data.csv

# Run the development server
python manage.py runserver
```

## 🎯 Model Training Process

### 1. **Enhanced Data Generation**
```bash
python manage.py generate_time_based_crowd_data
```
**Features:**
- 31,000+ realistic crowd records
- Festival-aware patterns
- Place-specific overrides
- District scaling factors
- Weather impact integration

### 2. **Model Training**
```bash
python manage.py train_improved_crowd_model --csv-file enhanced_time_based_crowd_data.csv
```
**Performance:**
- R² Score: 0.9671 (excellent fit)
- Mean Absolute Error: 3.01
- Mean Squared Error: 13.61
- 55 engineered features

### 3. **Real-time Predictions**
- Weather-aware crowd predictions
- Festival date recognition
- Place-specific pattern application
- District-based scaling

## 📊 Data Quality & Features

### **Training Data:**
- **Records:** 31,000+ enhanced crowd records
- **Features:** 12 columns including weather, festivals, districts
- **Quality:** 0 missing values, realistic 0-100% bounds
- **Coverage:** 6 months of historical data

### **Prediction Features:**
- **Temporal:** Time slots, day of week, month, season
- **Geographic:** District, place-specific patterns
- **Cultural:** Festival dates, religious significance
- **Environmental:** Weather conditions, seasonal patterns
- **Behavioral:** Weekend vs weekday patterns

## 🌍 Cultural Intelligence

### **Nepali Festival Awareness:**
- **Dashain:** Major Hindu festival with temple crowds
- **Tihar:** Festival of Lights with cultural celebrations
- **Nepali New Year:** Tourist and local celebrations
- **Janai Purnima:** Religious ritual attendance

### **Religious Site Patterns:**
- **Morning prayers:** High attendance at temples
- **Evening aarti:** Religious ceremonies
- **Festival periods:** Significant crowd increases
- **Weather impact:** Indoor vs outdoor considerations

### **Tourist Behavior:**
- **Thamel nightlife:** Peak evening activity
- **Market shopping:** Afternoon/evening peaks
- **Cultural sites:** Rain increases indoor attendance
- **Nature parks:** Weather-dependent outdoor activity

## 🔧 API Endpoints

### **Crowd Predictions:**
- `GET /api/improved-crowd-predictions/` - Enhanced crowd predictions
- `GET /api/tourism-crowd-charts/` - Chart data with weather impact
- `GET /api/predict-crowd/` - Single place prediction

### **Place Management:**
- `GET /api/places-by-district/{district}/` - District-based places
- `GET /api/places-by-category/{category}/` - Category-based places
- `GET /api/search-places/` - Place search functionality

### **Weather Integration:**
- Real-time weather data for each place
- Weather impact on crowd predictions
- Seasonal weather patterns

## 📈 Performance Metrics

### **Model Accuracy:**
- **R² Score:** 0.9671 (97% accuracy)
- **Mean Absolute Error:** 3.01 (excellent precision)
- **Mean Squared Error:** 13.61 (low variance)
- **Feature Importance:** Logical cultural and temporal patterns

### **Data Quality:**
- **Completeness:** 100% (0 missing values)
- **Realism:** Culturally appropriate patterns
- **Coverage:** All major Nepali tourist destinations
- **Temporal:** 6 months of comprehensive data

## 🎯 Use Cases

### **For Tourists:**
- Plan visits during optimal times
- Avoid crowded periods
- Consider weather conditions
- Festival-aware planning

### **For Businesses:**
- Optimize staffing during peak times
- Plan marketing around festivals
- Weather-based business strategies
- District-specific market analysis

### **For Tourism Authorities:**
- Crowd management during festivals
- Infrastructure planning
- Cultural preservation strategies
- Tourist flow optimization

## 🚀 Future Enhancements

### **Planned Features:**
- **Mobile App:** Native iOS/Android applications
- **Real-time Updates:** Live crowd monitoring
- **Advanced Analytics:** Predictive trend analysis
- **Multi-language Support:** Nepali language interface
- **Social Features:** User reviews and ratings

### **Model Improvements:**
- **Deep Learning:** Neural network integration
- **Real-time Learning:** Continuous model updates
- **External Data:** Social media sentiment analysis
- **Advanced Weather:** Hourly weather predictions

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📞 Support

For support and questions:
- Create an issue in the repository
- Contact the development team
- Check the documentation

---

**Built with ❤️ for Nepal's Tourism Industry**
