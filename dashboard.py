import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# ---- Load Model and Data ----
try:
    model = joblib.load("weather_prediction_model.pkl")
except FileNotFoundError:
    print("ERROR: Run weather_forecasting.py first to generate the model file.")
    exit()

try:
    df = pd.read_csv("synthetic_weather_data.csv")
except FileNotFoundError:
    print("ERROR: Run weather_forecasting.py first to generate the dataset.")
    exit()

# ---- Input Validation ----
def validate_inputs(humidity, wind, pressure, precip):
    checks = [
        (humidity,  0,    100,  "Humidity must be between 0 and 100 %"),
        (wind,      0,    200,  "Wind speed must be between 0 and 200 km/h"),
        (pressure,  800,  1200, "Pressure must be between 800 and 1200 hPa"),
        (precip,    0,    500,  "Precipitation must be between 0 and 500 mm"),
    ]
    for val, low, high, msg in checks:
        if not (low <= val <= high):
            return False, msg
    return True, ""

# ---- GUI Window ----
root = tk.Tk()
root.title("Weather Prediction ML Dashboard")
root.geometry("1100x700")
root.configure(bg="#1e1e1e")

style = ttk.Style()
style.theme_use("clam")

# ---- Title ----
tk.Label(
    root,
    text="Weather Prediction Dashboard",
    font=("Segoe UI", 22, "bold"),
    bg="#1e1e1e",
    fg="white"
).pack(pady=10)

# ---- Main Frame ----
main_frame = ttk.Frame(root)
main_frame.pack(fill="both", expand=True, padx=20, pady=10)

# ---- Input Frame ----
input_frame = ttk.LabelFrame(main_frame, text="Input Weather Parameters")
input_frame.grid(row=0, column=0, padx=10, pady=10, sticky="n")

humidity_var  = tk.DoubleVar(value=65.0)
wind_var      = tk.DoubleVar(value=10.0)
pressure_var  = tk.DoubleVar(value=1010.0)
precip_var    = tk.DoubleVar(value=10.0)

fields = [
    ("Humidity (%)",       humidity_var,  "Range: 0 - 100"),
    ("Wind Speed (km/h)",  wind_var,      "Range: 0 - 200"),
    ("Pressure (hPa)",     pressure_var,  "Range: 800 - 1200"),
    ("Precipitation (mm)", precip_var,    "Range: 0 - 500"),
]

for i, (label, var, hint) in enumerate(fields):
    ttk.Label(input_frame, text=label).grid(
        row=i, column=0, pady=6, sticky="w", padx=5
    )
    ttk.Entry(input_frame, textvariable=var, width=12).grid(
        row=i, column=1, padx=5
    )
    ttk.Label(input_frame, text=hint, foreground="gray").grid(
        row=i, column=2, padx=5
    )

# ---- Result Label ----
result_label = tk.Label(
    input_frame,
    text="Predicted Temperature: -- C",
    font=("Segoe UI", 14, "bold"),
    fg="orange",
    bg="#ffffff"
)
result_label.grid(row=5, columnspan=3, pady=15)

# ---- Graph Frame ----
graph_frame = ttk.LabelFrame(main_frame, text="Weather Visualization")
graph_frame.grid(row=0, column=1, padx=10, pady=10)

fig, ax = plt.subplots(figsize=(6, 5), dpi=100)
fig.patch.set_facecolor("#2e2e2e")
ax.set_facecolor("#2e2e2e")

canvas = FigureCanvasTkAgg(fig, master=graph_frame)
canvas.get_tk_widget().pack()

# ---- Predict Function ----
def predict_weather():
    try:
        humidity  = humidity_var.get()
        wind      = wind_var.get()
        pressure  = pressure_var.get()
        precip    = precip_var.get()

        valid, error_msg = validate_inputs(humidity, wind, pressure, precip)
        if not valid:
            messagebox.showerror("Invalid Input", error_msg)
            return

        input_data = pd.DataFrame(
            [[humidity, wind, pressure, precip]],
            columns=["humidity", "wind_speed", "pressure", "precipitation"]
        )

        temp = round(model.predict(input_data)[0], 2)
        result_label.config(text=f"Predicted Temperature: {temp:.2f} C")

        ax.cla()
        parameters = ["Humidity", "Wind", "Pressure", "Precip", "Temp"]
        values     = [humidity, wind, pressure, precip, temp]
        colors     = ["#378ADD", "#378ADD", "#378ADD", "#378ADD", "#E24B4A"]

        bars = ax.bar(parameters, values, color=colors, edgecolor="none")
        ax.set_title("Live Weather Parameter Comparison", color="white", pad=10)
        ax.set_ylabel("Values", color="white")
        ax.tick_params(colors="white")
        ax.set_facecolor("#2e2e2e")

        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1,
                f"{val:.1f}",
                ha="center", va="bottom",
                color="white", fontsize=9
            )

        fig.patch.set_facecolor("#2e2e2e")
        canvas.draw()

    except tk.TclError:
        messagebox.showerror("Invalid Input", "Please enter valid numeric values.")
    except Exception as e:
        messagebox.showerror("Error", f"Unexpected error: {str(e)}")

# ---- Clear Function ----
def clear_data():
    humidity_var.set(65.0)
    wind_var.set(10.0)
    pressure_var.set(1010.0)
    precip_var.set(10.0)
    result_label.config(text="Predicted Temperature: -- C")
    ax.cla()
    ax.set_facecolor("#2e2e2e")
    canvas.draw()

# ---- Buttons ----
button_frame = ttk.Frame(input_frame)
button_frame.grid(row=6, columnspan=3, pady=10)

ttk.Button(button_frame, text="Predict", command=predict_weather).grid(
    row=0, column=0, padx=5
)
ttk.Button(button_frame, text="Clear", command=clear_data).grid(
    row=0, column=1, padx=5
)

# ---- Historical Trend Window ----
def show_history():
    history_window = tk.Toplevel(root)
    history_window.title("Historical Weather Trends")
    history_window.configure(bg="#1e1e1e")

    fig2, axes = plt.subplots(2, 1, figsize=(8, 6))
    fig2.patch.set_facecolor("#2e2e2e")

    axes[0].plot(df["temperature"].head(100), color="#E24B4A", linewidth=1.5)
    axes[0].set_title("Historical Temperature Trend", color="white")
    axes[0].set_ylabel("Temperature C", color="white")
    axes[0].set_facecolor("#2e2e2e")
    axes[0].tick_params(colors="white")

    axes[1].plot(df["humidity"].head(100), color="#378ADD", linewidth=1.5)
    axes[1].set_title("Historical Humidity Trend", color="white")
    axes[1].set_xlabel("Days", color="white")
    axes[1].set_ylabel("Humidity %", color="white")
    axes[1].set_facecolor("#2e2e2e")
    axes[1].tick_params(colors="white")

    fig2.tight_layout()

    canvas2 = FigureCanvasTkAgg(fig2, master=history_window)
    canvas2.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=10)
    canvas2.draw()

# ---- Show History Button ----
ttk.Button(root, text="Show Historical Data", command=show_history).pack(pady=5)

# ---- Status Bar ----
tk.Label(
    root,
    text="Model: Random Forest (100 estimators)  |  Training samples: 800  |  Ready",
    font=("Segoe UI", 9),
    bg="#111111",
    fg="#888888",
    anchor="w"
).pack(side="bottom", fill="x", padx=10, pady=4)

root.mainloop()
