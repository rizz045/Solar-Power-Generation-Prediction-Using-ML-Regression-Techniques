import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

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

# Load model, scaler, and features
with open('random_forest_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('features.pkl', 'rb') as f:
    feature_names = pickle.load(f)
# Function to make predictions
def predict_power(input_data):
    input_df = pd.DataFrame([input_data], columns=feature_names)
    dummy_row = input_df.copy()
    dummy_row['power_generated'] = 0
    scaled_data = scaler.transform(dummy_row)
    input_scaled = scaled_data[:, :-1]
    log_prediction = model.predict(input_scaled)
    prediction = np.exp(log_prediction)[0]
    return prediction

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
    distance_to_solar_noon = st.slider("Distance to Solar Noon (degrees)", -180.0, 180.0, 0.0, 0.1)
    temperature = st.slider("Temperature (°C)", -20.0, 50.0, 25.0, 0.1)
    wind_speed = st.slider("Wind Speed (km/h)", 0.0, 100.0, 10.0, 0.1)
    sky_cover = st.slider("Sky Cover (oktas)", 0.0, 8.0, 2.0, 0.1)

with col2:
    st.subheader("Atmospheric Conditions")
    wind_direction = st.slider("Wind Direction (degrees)", 0.0, 360.0, 180.0, 1.0)
    visibility = st.slider("Visibility (km)", 0.0, 50.0, 10.0, 0.1)
    humidity = st.slider("Humidity (%)", 0.0, 100.0, 50.0, 0.1)
    average_pressure = st.slider("Average Pressure (hPa)", 800.0, 1100.0, 1013.0, 0.1)

input_data = {
    'distance_to_solar_noon': distance_to_solar_noon,
    'temperature': temperature,
    'wind_direction': wind_direction,
    'wind_speed': wind_speed,
    'sky_cover': sky_cover,
    'visibility': visibility,
    'humidity': humidity,
    'average_pressure': average_pressure
}

missing_features = set(feature_names) - set(input_data.keys()) - {'power_generated'}
if missing_features:
    st.error(f"Missing features in input: {missing_features}")

if st.button('Predict Power Generation'):
    try:
        prediction = predict_power(input_data)
        st.success(f"Predicted Power Generation: {prediction:.2f} MW")

        st.subheader("Prediction Results")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Power Output", f"{prediction:.2f} MW")
        with col2:
            efficiency = (prediction / 10) * 100
            st.metric("Estimated Efficiency", f"{efficiency:.1f}%")
        with col3:
            status = "Low ☁️" if prediction < 2 else "Moderate ⛅" if prediction < 5 else "High ☀️"
            st.metric("Generation Status", status)

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

st.sidebar.header("About")
st.sidebar.info("This predictive model uses a Random Forest algorithm trained on solar power data.")

st.sidebar.header("Model Features")
st.sidebar.write(feature_names)
