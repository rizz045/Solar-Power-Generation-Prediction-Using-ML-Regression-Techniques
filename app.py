import streamlit as st
import joblib
import numpy as np
import time 
import pandas as pd

# Load Model, Encoder & Scaler
model = joblib.load('solar_power_generation_xgbr_model.pkl')

# Custom Styling

st.set_page_config(page_title="Solar Power Generation Predictor", layout="centered")
st.image("https://cdn.shopify.com/s/files/1/0493/9834/9974/files/can-solar-generators-power-a-calculator.jpg?v=1685590896")

# App Title
st.markdown("<h1>Solar Power Generation Predictor</h1>", unsafe_allow_html=True)

# Input Section
st.markdown("---")
st.markdown("### 🌤️ Input Environmental Conditions")
st.markdown("Use the sliders and number inputs below to provide current weather conditions for solar power prediction.")

with st.container():
    col1, col2 = st.columns(2)

    with col1:
        dist = st.number_input("📍 Distance to Solar Noon", min_value=0.0, max_value=1.5, step=0.1, format='%.4f')
        wind_speed = st.slider("🌬️ Wind Speed (mph)", min_value=1.1, max_value=22.1, step=0.1, format='%.1f')
        sky_cover = st.slider("☁️ Sky Cover", min_value=0, max_value=4, step=1)
        avg_wind_speed = st.slider("📏 Avg Wind Speed (Period)", min_value=0.0, max_value=30.0, step=1.0)

    with col2:
        temp = st.number_input("🌡️ Temperature (°C)", min_value=42, max_value=78, step=1)
        wind_direction = st.number_input("🧭 Wind Direction (1-32)", min_value=1, max_value=32, step=1)
        humidity = st.number_input("💧 Humidity (%)", min_value=0, max_value=100, step=1)
        avg_pressure = st.slider("🔽 Avg Pressure (inHg)", min_value=29.64, max_value=30.39, step=0.1)

st.markdown("---")


# Convert Inputs
wind_dir_sine = np.sin(2 * np.pi * wind_direction / 360 )
wind_dir_cosine = np.cos(2 * np.pi * wind_direction / 360 )

# Define input in dictionary format
input_dict = {
    'distance-to-solar-noon': [dist],
    'temperature': [temp],
    'wind-speed': [wind_speed],
    'sky-cover': [sky_cover],
    'humidity': [humidity],
    'average-wind-speed-(period)': [avg_wind_speed],
    'average-pressure-(period)': [avg_pressure],
    'wind_dir_sin': [wind_dir_sine],
    'wind_dir_cos': [wind_dir_cosine]
}

# Convert to DataFrame
input_df = pd.DataFrame(input_dict)

# Predict Button with Loading Animation
if st.button("🔍 Predict Power Generated"):
    with st.spinner("Analyzing environment data..."):
        time.sleep(2)  # Short delay for loading animation
        prediction = model.predict(input_df)

    # Display Prediction
    pred = abs(prediction[0])
    st.success(f"⚡ Estimated Power Output: **{pred:.2f} J**")

# Footer
st.markdown("---")
st.markdown("<h5 style='text-align:center;'>Do Visit Us Again ❤️</h5>", unsafe_allow_html=True)
