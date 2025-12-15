# Lagos Malaria Prevalence Predictor

A machine learning application that predicts malaria prevalence across Lagos State communities using real-time environmental data.

## Features

- Real-time malaria risk prediction
- Community-level geographic analysis
- Interactive map visualization
- Live environmental data fetching (temperature, rainfall, humidity, NDVI)
- Risk categorization (Low, Moderate, High)

## How It Works

The app uses a Random Forest machine learning model trained on:
- Environmental factors (NDVI, temperature, rainfall, humidity)
- Demographic data (population density)
- Temporal patterns (seasonal variations)

The model achieves R² = 0.71, explaining 71% of variance in malaria prevalence.

## Tech Stack

- **Python** - Core programming language
- **Streamlit** - Web application framework
- **scikit-learn** - Machine learning (Random Forest)
- **Plotly** - Interactive visualizations
- **Pandas/NumPy** - Data processing
- **Open-Meteo API** - Real-time weather data

## Usage

1. Select an LGA (Local Government Area)
2. Choose a community
3. Select year and month
4. Click "Fetch Live Data & Predict"
5. View prediction on map and risk assessment

## Data

Currently uses synthetic prevalence data as proof-of-concept. Framework is ready for real surveillance data from Lagos State health authorities.

## Disclaimer

This is a predictive model and should be used alongside actual malaria surveillance data. Predictions are estimates based on environmental correlates and should not replace clinical diagnosis or public health surveillance.

## Contact

Built by [Bernard Muoneme]

Connect:[www.linkedin.com/in/bernardmuoneme]|[bmuoneme@gmail.com]

## Acknowledgments


Environmental data provided by Open-Meteo Archive API.

