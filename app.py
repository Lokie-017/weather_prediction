import streamlit as st
import joblib
import pandas as pd

st.title("🌦️ Weather Predictor App")
st.subheader("Choose your model and enter input values")

# Model selector
model_option = st.selectbox("Select model:", ["Temperature (Regression)", "Precip Type (Classification)"])

# Input fields (must match your training features)
summary = st.number_input("Summary (encoded)", min_value=0)
apparent_temp = st.number_input("Apparent Temperature (C)")
humidity = st.slider("Humidity", 0.0, 1.0, 0.7)
wind_speed = st.number_input("Wind Speed (km/h)")
visibility = st.number_input("Visibility (km)")
pressure = st.number_input("Pressure (millibars)")
year = st.selectbox("Year", [2016])
month = st.selectbox("Month", list(range(1,13)))
day = st.selectbox("Day", list(range(1,32)))
hour = st.selectbox("Hour", list(range(0,24)))
daily_summary = st.number_input("Daily Summary (encoded)", min_value=0)

# Prediction button
if st.button("Predict"):
    input_data = {
        "Summary": summary,
        "Apparent Temperature (C)": apparent_temp,
        "Humidity": humidity,
        "Wind Speed (km/h)": wind_speed,
        "Visibility (km)": visibility,
        "Pressure (millibars)": pressure,
        "Year": year,
        "Month": month,
        "Day": day,
        "Hour": hour,
        "Daily Summary": daily_summary
    }

    df_input = pd.DataFrame([input_data])

    if model_option == "Temperature (Regression)":
        
        

        model = joblib.load("models/regression_randomforest.pkl")
        prediction = model.predict(df_input)[0]
        st.success(f"🌡️ Predicted Temperature: {prediction:.2f} °C")
    
    elif model_option == "Precip Type (Classification)":
        model = joblib.load("models/classification_randomforest.pkl")
        prediction = model.predict(df_input)[0]
        label_map = {0: "Rain", 1: "Snow", 2: "Unknown"}
        st.success(f"☁️ Predicted Precip Type: {label_map[prediction]}")
