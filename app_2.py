import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ✅ MUST BE FIRST Streamlit command!
st.set_page_config(page_title="Solar Power Generation Predictor", layout="wide")

# ✅ Styling - Yellow & Black Theme
st.markdown("""
    <style>
        .main {
            background-color: #000000;
            color: #FFD700;
        }
        h1, h2, h3 {
            color: #FFD700 !important;
        }
        .stButton>button {
            background-color: #FFD700;
            color: black;
            font-weight: bold;
        }
        .stMetric label {
            color: #FFD700 !important;
        }
        .stSlider .st-cz {
            color: #FFD700;
        }
        .block-container {
            background-color: #000000;
        }
    </style>
""", unsafe_allow_html=True)

# Load the model
model = joblib.load('solar_power_generation_xgbr_model.pkl')

# Streamlit app
#st.set_page_config(page_title="Solar Power Generation Predictor", layout="wide")

st.markdown("<h1 style='text-align: center;'>☀️ Solar Power Generation Prediction</h1>", unsafe_allow_html=True)

st.markdown("""
This app predicts solar power generation based on weather and environmental conditions.
Adjust the input parameters using the sliders and see the predicted power output.
""")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Solar Position & Weather")
    dist = st.number_input("📍 Distance to Solar Noon", min_value=0.0, max_value=1.5, step=0.1, format='%.4f')
    wind_speed = st.slider("🌬️ Wind Speed (mph)", min_value=1.1, max_value=22.1, step=0.1, format='%.1f')
    sky_cover = st.slider("☁️ Sky Cover", min_value=0, max_value=4, step=1)
    avg_wind_speed = st.slider("📏 Avg Wind Speed (Period)", min_value=0.0, max_value=30.0, step=1.0)

with col2:
    temp = st.number_input("🌡️ Temperature (°C)", min_value=42, max_value=78, step=1)
    wind_direction = st.number_input("🧭 Wind Direction (1-32)", min_value=1, max_value=32, step=1)
    humidity = st.number_input("💧 Humidity (%)", min_value=0, max_value=100, step=1)
    avg_pressure = st.slider("🔽 Avg Pressure (inHg)", min_value=29.64, max_value=30.39, step=0.1)

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


if st.button('Predict Power Generation'):
    try:
        # Make predictions
        prediction = model.predict(input_df)[0]
        st.success(f"Predicted Power Generation: {prediction:.2f} MW")

        st.subheader("Prediction Results")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Power Output", f"{prediction:.2f} MW")
        with col2:
            efficiency = (prediction / 33500) * 100
            st.metric("Estimated Efficiency", f"{efficiency:.1f}%")
        with col3:
            status = "Low ☁️" if prediction < 10000 else "Moderate ⛅" if prediction < 20000 else "High ☀️"
            st.metric("Generation Status", status)

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

st.sidebar.header("About")
st.sidebar.info("This predictive model uses a XGBRegressor algorithm trained on solar power data.")

feature_names = ['Distance', 'Temperature', 'Wind Direction','Wind Speed', 'Sky Cover','Humidity','Average Wind Speed','Average Pressure']
st.sidebar.header("Model Features")
st.sidebar.write(feature_names)
