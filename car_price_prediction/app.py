import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(BASE_DIR, 'car_price_model.pkl')
model = pickle.load(open(model_path, 'rb'))


st.title("🚗 Car Price Prediction App")
st.write("Predict the price of a used Ford car based on its features")

st.sidebar.header("Enter Car Details:")

year = st.sidebar.number_input("Year", min_value=1990, max_value=2025, value=2018)
mileage = st.sidebar.number_input("Mileage (in miles)", min_value=0, max_value=200000, value=30000)
tax = st.sidebar.number_input("Tax (£)", min_value=0, max_value=600, value=150)
mpg = st.sidebar.number_input("Miles per gallon (MPG)", min_value=0.0, max_value=200.0, value=50.0)
engineSize = st.sidebar.number_input("Engine Size (in L)", min_value=0.0, max_value=10.0, value=1.5)

model_name = st.sidebar.selectbox("Model", ['Fiesta', 'Focus', 'Kuga', 'Mondeo', 'Puma', 'EcoSport'])
transmission = st.sidebar.selectbox("Transmission", ['Manual', 'Automatic', 'Semi-Auto'])
fuelType = st.sidebar.selectbox("Fuel Type", ['Petrol', 'Diesel', 'Hybrid', 'Electric'])

input_data = pd.DataFrame({
    'model': [model_name],
    'year': [year],
    'transmission': [transmission],
    'mileage': [mileage],
    'fuelType': [fuelType],
    'tax': [tax],
    'mpg': [mpg],
    'engineSize': [engineSize]
})

st.subheader("📋 Input Summary")
st.write(input_data)

input_data_encoded = pd.get_dummies(input_data)

try:
    train_columns = pickle.load(open('train_columns.pkl', 'rb'))
    input_data_encoded = input_data_encoded.reindex(columns=train_columns, fill_value=0)
except:
    st.warning("⚠️ Missing 'train_columns.pkl'. Please save your training columns from your notebook.")

if st.button("Predict Price"):
    try:
        prediction = model.predict(input_data_encoded)
        st.success(f"💰 Estimated Car Price: £{prediction[0]:,.2f}")
    except Exception as e:
        st.error(f"Error: {e}")


