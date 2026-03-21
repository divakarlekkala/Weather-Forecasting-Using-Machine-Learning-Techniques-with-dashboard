import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

st.set_page_config(
    page_title="Weather Forecasting ML",
    page_icon="🌤",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .block-container { padding-top: 2rem; }
    .metric-card {
        background: #1e2130;
        border: 1px solid #2e3250;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        text-align: center;
    }
    .metric-card .label {
        font-size: 12px;
        color: #8b92a5;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 6px;
    }
    .metric-card .value {
        font-size: 28px;
        font-weight: 700;
        color: #ffffff;
    }
    .metric-card .unit { font-size: 13px; color: #8b92a5; margin-top: 2px; }
    .predict-result {
        background: #1a2a4a;
        border: 1px solid #2a5298;
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        margin: 1rem 0;
    }
    .predict-result .temp-value {
        font-size: 64px;
        font-weight: 800;
        color: #ff6b35;
        line-height: 1;
    }
    .predict-result .temp-label { font-size: 16px; color: #8b92a5; margin-top: 8px; }
    .section-title {
        font-size: 18px;
        font-weight: 600;
        color: #ffffff;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid #2e3250;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def train_model():
    np.random.seed(42)
    num_days = 1000
    data = {
        "date":          pd.date_range(start="2020-01-01", periods=num_days, freq="D"),
        "temperature":   np.random.uniform(10, 40, num_days),
        "humidity":      np.random.uniform(20, 90, num_days),
        "wind_speed":    np.random.uniform(0, 20, num_days),
        "pressure":      np.random.uniform(900, 1100, num_days),
        "precipitation": np.random.uniform(0, 50, num_days),
    }
    df = pd.DataFrame(data)
    FEATURES = ["humidity", "wind_speed", "pressure", "precipitation"]
    X = df[FEATURES]
    y = df["temperature"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    metrics = {
        "r2":   round(r2_score(y_test, y_pred), 3),
        "mae":  round(mean_absolute_error(y_test, y_pred), 3),
        "rmse": round(np.sqrt(mean_squared_error(y_test, y_pred)), 3),
    }
    importance_df = pd.DataFrame({
        "Feature":    FEATURES,
        "Importance": model.feature_importances_
    }).sort_values("Importance", ascending=False)
    actual_vs_pred = pd.DataFrame({
        "Actual":    y_test.values[:60],
        "Predicted": y_pred[:60]
    })
    return model, df, metrics, importance_df, actual_vs_pred

model, df, metrics, importance_df, actual_vs_pred = train_model()

def make_dark_fig(figsize=(6, 3)):
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("#1e2130")
    ax.set_facecolor("#1e2130")
    ax.tick_params(colors="white")
    ax.spines[:].set_visible(False)
    return fig, ax

with st.sidebar:
    st.markdown("## Weather Inputs")
    st.caption("Adjust parameters to predict temperature")
    st.divider()
    humidity = st.slider("Humidity (%)",        0.0,   100.0, 65.0,  0.5)
    wind     = st.slider("Wind Speed (km/h)",   0.0,   20.0,  10.0,  0.5)
    pressure = st.slider("Pressure (hPa)",      900.0, 1100.0,1010.0,1.0)
    precip   = st.slider("Precipitation (mm)",  0.0,   50.0,  10.0,  0.5)
    st.divider()
    predict_btn = st.button("Predict Temperature", use_container_width=True)
    st.divider()
    st.markdown("**Model Performance**")
    st.metric("R2 Score", metrics["r2"])
    st.metric("MAE",  f"{metrics['mae']} C")
    st.metric("RMSE", f"{metrics['rmse']} C")
    st.divider()
    st.caption("Random Forest · 100 estimators · 800 training samples")

st.markdown("# Weather Forecasting ML Dashboard")
st.caption("Predict temperature using a Random Forest model trained on 1,000 days of weather data.")
st.divider()

tab1, tab2, tab3 = st.tabs(["Predict", "Historical Trends", "Model Insights"])

with tab1:
    st.markdown('<div class="section-title">Current Input Parameters</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f'<div class="metric-card"><div class="label">Humidity</div><div class="value">{humidity:.0f}</div><div class="unit">%</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="metric-card"><div class="label">Wind Speed</div><div class="value">{wind:.1f}</div><div class="unit">km/h</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="metric-card"><div class="label">Pressure</div><div class="value">{pressure:.0f}</div><div class="unit">hPa</div></div>', unsafe_allow_html=True)
    with c4:
        st.markdown(f'<div class="metric-card"><div class="label">Precipitation</div><div class="value">{precip:.1f}</div><div class="unit">mm</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    if predict_btn:
        input_data = pd.DataFrame(
            [[humidity, wind, pressure, precip]],
            columns=["humidity", "wind_speed", "pressure", "precipitation"]
        )
        temp = round(model.predict(input_data)[0], 2)

        st.markdown(f"""
        <div class="predict-result">
            <div class="temp-value">{temp:.1f}°C</div>
            <div class="temp-label">Predicted Temperature</div>
        </div>
        """, unsafe_allow_html=True)

        if temp < 15:
            st.info("Cold weather conditions")
        elif temp < 25:
            st.success("Pleasant weather conditions")
        elif temp < 33:
            st.warning("Warm weather conditions")
        else:
            st.error("Hot weather conditions")

        st.divider()
        st.markdown('<div class="section-title">Parameter Breakdown</div>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            fig1, ax1 = make_dark_fig()
            params = ["Humidity (%)", "Wind (km/h)", "Precip (mm)"]
            vals   = [humidity, wind, precip]
            colors = ["#378ADD", "#5DCAA5", "#7F77DD"]
            bars = ax1.bar(params, vals, color=colors, edgecolor="none", width=0.5)
            ax1.set_title("Weather Inputs", color="white", pad=10, fontsize=12)
            ax1.set_ylabel("Value", color="white")
            for bar, val in zip(bars, vals):
                ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                         f"{val:.1f}", ha="center", va="bottom", color="white", fontsize=10)
            fig1.patch.set_facecolor("#1e2130")
            st.pyplot(fig1)

        with col2:
            fig2, ax2 = make_dark_fig()
            gauge_labels = ["Min (10C)", f"Predicted ({temp:.1f}C)", "Max (40C)"]
            gauge_vals   = [10, temp, 40]
            gauge_colors = ["#378ADD", "#ff6b35", "#E24B4A"]
            bars2 = ax2.bar(gauge_labels, gauge_vals, color=gauge_colors, edgecolor="none", width=0.5)
            ax2.set_title("Temperature Scale", color="white", pad=10, fontsize=12)
            ax2.set_ylabel("Celsius", color="white")
            for bar, val in zip(bars2, gauge_vals):
                ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
                         f"{val:.1f}C", ha="center", va="bottom", color="white", fontsize=10)
            fig2.patch.set_facecolor("#1e2130")
            st.pyplot(fig2)

    else:
        st.markdown("""
        <div style="text-align:center;padding:3rem;color:#8b92a5;">
            <div style="font-size:48px;">🔮</div>
            <div style="font-size:18px;margin-top:1rem;">Adjust sliders and click Predict Temperature</div>
            <div style="font-size:14px;margin-top:0.5rem;">Results will appear here</div>
        </div>
        """, unsafe_allow_html=True)

with tab2:
    st.markdown('<div class="section-title">Historical Weather Data</div>', unsafe_allow_html=True)
    days = st.slider("Show last N days", 50, 1000, 100, 50)
    hist = df.tail(days)

    col1, col2 = st.columns(2)
    trend_data = [
        ("temperature", "Temperature Trend", "#ff6b35", "C"),
        ("humidity",    "Humidity Trend",    "#378ADD", "%"),
        ("wind_speed",  "Wind Speed Trend",  "#5DCAA5", "km/h"),
        ("precipitation","Precipitation Trend","#7F77DD","mm"),
    ]
    for i, (col, title, color, unit) in enumerate(trend_data):
        with (col1 if i % 2 == 0 else col2):
            fig, ax = make_dark_fig()
            ax.plot(hist[col].values, color=color, linewidth=1.5)
            ax.fill_between(range(len(hist)), hist[col].values, alpha=0.2, color=color)
            ax.set_title(title, color="white", fontsize=12)
            ax.set_ylabel(unit, color="white")
            fig.patch.set_facecolor("#1e2130")
            st.pyplot(fig)

    st.divider()
    st.markdown('<div class="section-title">Dataset Summary</div>', unsafe_allow_html=True)
    st.dataframe(
        df[["temperature","humidity","wind_speed","pressure","precipitation"]].describe().round(2),
        use_container_width=True
    )

with tab3:
    st.markdown('<div class="section-title">Model Performance</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("R2 Score",   metrics["r2"])
    c2.metric("MAE",        f"{metrics['mae']} C")
    c3.metric("RMSE",       f"{metrics['rmse']} C")
    c4.metric("Train Size", "800 samples")

    st.divider()
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-title">Feature Importance</div>', unsafe_allow_html=True)
        fig7, ax7 = make_dark_fig(figsize=(6, 4))
        colors7 = ["#ff6b35", "#378ADD", "#5DCAA5", "#7F77DD"]
        bars7 = ax7.barh(importance_df["Feature"], importance_df["Importance"],
                         color=colors7, edgecolor="none")
        ax7.set_title("Which features matter most?", color="white", fontsize=12)
        for bar, val in zip(bars7, importance_df["Importance"]):
            ax7.text(val+0.002, bar.get_y()+bar.get_height()/2,
                     f"{val:.3f}", va="center", color="white", fontsize=10)
        fig7.patch.set_facecolor("#1e2130")
        st.pyplot(fig7)

    with col2:
        st.markdown('<div class="section-title">Actual vs Predicted</div>', unsafe_allow_html=True)
        fig8, ax8 = make_dark_fig(figsize=(6, 4))
        ax8.plot(actual_vs_pred["Actual"],    color="#5DCAA5", linewidth=1.5, label="Actual")
        ax8.plot(actual_vs_pred["Predicted"], color="#ff6b35", linewidth=1.5, linestyle="--", label="Predicted")
        ax8.set_title("First 60 test samples", color="white", fontsize=12)
        ax8.set_ylabel("Temperature C", color="white")
        ax8.legend(facecolor="#2e3250", labelcolor="white")
        fig8.patch.set_facecolor("#1e2130")
        st.pyplot(fig8)

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **Algorithm:** Random Forest Regressor

        **Why Random Forest?**
        - Handles non-linear relationships well
        - Resistant to overfitting
        - Provides feature importance scores
        - Works well with small datasets
        """)
    with col2:
        st.markdown("""
        **Training Configuration**
        - Estimators: 100 decision trees
        - Test split: 20% (200 samples)
        - Train split: 80% (800 samples)
        - Random state: 42 (reproducible)
        """)

st.divider()
st.caption("Built by Lekkala Divakar  |  linkedin.com/in/lekkaladivakar  |  github.com/divakarlekkala")
