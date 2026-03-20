# ============================================================
# STEP 1 — Generate Synthetic Dataset
# ============================================================
import pandas as pd
import numpy as np

num_days = 1000
np.random.seed(42)

data = {
    "date": pd.date_range(start="2020-01-01", periods=num_days, freq="D"),
    "temperature": np.random.uniform(10, 40, num_days),
    "humidity": np.random.uniform(20, 90, num_days),
    "wind_speed": np.random.uniform(0, 20, num_days),
    "pressure": np.random.uniform(900, 1100, num_days),
    "precipitation": np.random.uniform(0, 50, num_days),
}

df = pd.DataFrame(data)
df.to_csv("synthetic_weather_data.csv", index=False)
print(f"Dataset created: {len(df)} rows, {df.shape[1]} columns")
print(df.describe().round(2))


# ============================================================
# STEP 2 — Train Model and Evaluate
# ============================================================
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib

df = pd.read_csv("synthetic_weather_data.csv")

FEATURES = ["humidity", "wind_speed", "pressure", "precipitation"]
TARGET = "temperature"

X = df[FEATURES]
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

r2   = round(r2_score(y_test, y_pred), 4)
mae  = round(mean_absolute_error(y_test, y_pred), 4)
rmse = round(np.sqrt(mean_squared_error(y_test, y_pred)), 4)

print("=" * 40)
print("MODEL EVALUATION")
print("=" * 40)
print(f"R2 Score : {r2}   (1.0 = perfect)")
print(f"MAE      : {mae} degrees C")
print(f"RMSE     : {rmse} degrees C")
print("=" * 40)

joblib.dump(model, "weather_prediction_model.pkl")
print("Model saved: weather_prediction_model.pkl")


# ============================================================
# STEP 3 — Feature Importance Chart
# ============================================================
import matplotlib.pyplot as plt

importances = model.feature_importances_
feat_df = pd.DataFrame({
    "Feature": FEATURES,
    "Importance": importances
}).sort_values("Importance", ascending=True)

fig, ax = plt.subplots(figsize=(7, 4))
ax.barh(feat_df["Feature"], feat_df["Importance"], color="#378ADD")
ax.set_title("Feature Importance - Weather Prediction Model")
ax.set_xlabel("Importance Score")
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=150)
plt.show()
print("Chart saved: feature_importance.png")


# ============================================================
# STEP 4 — Actual vs Predicted Chart
# ============================================================
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(y_test.values[:50], label="Actual", color="#1D9E75", linewidth=1.5)
ax.plot(y_pred[:50], label="Predicted", color="#E24B4A", linewidth=1.5, linestyle="--")
ax.set_title("Actual vs Predicted Temperature (first 50 test samples)")
ax.set_xlabel("Sample Index")
ax.set_ylabel("Temperature (degrees C)")
ax.legend()
plt.tight_layout()
plt.savefig("actual_vs_predicted.png", dpi=150)
plt.show()
print("Chart saved: actual_vs_predicted.png")


# ============================================================
# STEP 5 — Prediction Function
# ============================================================
model = joblib.load("weather_prediction_model.pkl")

def predict_temperature(humidity, wind_speed, pressure, precipitation):
    """
    Predict temperature given weather parameters.

    Parameters:
        humidity (float): Humidity percentage (20-90)
        wind_speed (float): Wind speed in km/h (0-20)
        pressure (float): Atmospheric pressure in hPa (900-1100)
        precipitation (float): Precipitation in mm (0-50)

    Returns:
        float: Predicted temperature in Celsius
    """
    input_data = pd.DataFrame(
        [[humidity, wind_speed, pressure, precipitation]],
        columns=["humidity", "wind_speed", "pressure", "precipitation"]
    )
    predicted_temp = model.predict(input_data)[0]
    return round(predicted_temp, 2)


print("Example Predictions:")
print(f"  Humid day     : {predict_temperature(80, 5, 1010, 20)} degrees C")
print(f"  Dry windy day : {predict_temperature(25, 15, 980, 2)} degrees C")
print(f"  Rainy day     : {predict_temperature(90, 10, 1000, 45)} degrees C")
