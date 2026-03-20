# Weather Forecasting Using Machine Learning

![Python](https://img.shields.io/badge/Python-3.9-blue)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.0-orange)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)
![Model](https://img.shields.io/badge/Model-Random%20Forest-green)

> A machine learning application that predicts temperature based on weather parameters — built with Python, Scikit-learn, and a Tkinter GUI dashboard.

---

## About the Project

This project uses a **Random Forest Regressor** to predict temperature (°C) based on real-world weather parameters like humidity, wind speed, atmospheric pressure, and precipitation.

The app includes a **desktop GUI dashboard** built with Tkinter that lets users input weather conditions and instantly see the predicted temperature along with a live bar chart visualization.

---

## Features

- **Temperature Prediction** — Predicts temperature in °C based on 4 input parameters
- **Interactive GUI Dashboard** — Desktop app built with Tkinter for real-time predictions
- **Live Bar Chart** — Visualizes input parameters vs predicted temperature instantly
- **Historical Trend Viewer** — Shows temperature and humidity trends across 100 days
- **Input Validation** — Checks all inputs are within valid real-world ranges
- **Model Evaluation** — R2 Score, MAE and RMSE calculated after training

---

## Tech Stack

- **Language:** Python 3.9
- **ML Library:** Scikit-learn (Random Forest Regressor)
- **Data Processing:** NumPy, Pandas
- **Visualization:** Matplotlib
- **GUI Framework:** Tkinter
- **Model Saving:** Joblib

---

## Project Structure

```
weather-forecasting-ml/
│
├── weather_forecasting.py       # Data generation, model training and evaluation
├── dashboard.py                 # Tkinter GUI dashboard
├── requirements.txt             # Project dependencies
└── README.md                    # Project documentation
```

---

## How It Works

```
Input Parameters
(Humidity, Wind Speed, Pressure, Precipitation)
        |
        v
Random Forest Regressor (100 estimators)
        |
        v
Predicted Temperature (°C)
        |
        v
Live Bar Chart + GUI Display
```

---

## Model Performance

| Metric | Value |
|--------|-------|
| Algorithm | Random Forest Regressor |
| Estimators | 100 trees |
| Training samples | 800 |
| Testing samples | 200 |
| Features | 4 (humidity, wind, pressure, precipitation) |

---

## Installation and Setup

1. Clone the repository
```
git clone https://github.com/divakarlekkala/weather-forecasting-ml
```

2. Install dependencies
```
pip install -r requirements.txt
```

3. Generate dataset and train the model
```
python weather_forecasting.py
```

4. Launch the dashboard
```
python dashboard.py
```

---

## Future Improvements

- Use real weather data from Open-Meteo API
- Convert desktop app to Streamlit web app for online access
- Add more weather features like cloud cover and UV index
- Support multi-city weather prediction

---

## Contact

**Lekkala Divakar**
- Email: lekkaladivakar2@gmail.com
- LinkedIn: linkedin.com/in/lekkaladivakar
- GitHub: github.com/divakarlekkala
- Live Project: https://jntuk-result-analyzer.streamlit.app
