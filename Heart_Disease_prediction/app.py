import streamlit as st
import requests

st.set_page_config(page_title="Heart Disease Predictor", layout="centered")
st.title("❤️ Heart Disease Prediction")

# -----------------------
# User input
# -----------------------
Age = st.number_input("Age", min_value=1, max_value=119, value=55)
Sex = st.selectbox("Sex", options=["M", "F"])
ChestPainType = st.selectbox("Chest Pain Type", options=["ATA", "NAP", "ASY", "TA"])
RestingBP = st.number_input("Resting Blood Pressure (mm Hg)", min_value=50, max_value=250, value=140)
Cholesterol = st.number_input("Cholesterol (mg/dl)", min_value=100, max_value=600, value=220)
FastingBS = st.selectbox("Fasting Blood Sugar >120 mg/dl?", options=[0, 1])
RestingECG = st.selectbox("Resting ECG", options=["Normal", "ST", "LVH"])
MaxHR = st.number_input("Maximum Heart Rate Achieved", min_value=60, max_value=249, value=150)
ExerciseAngina = st.selectbox("Exercise Induced Angina", options=["N", "Y"])
Oldpeak = st.number_input("Oldpeak (ST depression)", min_value=0.0, max_value=10.0, value=2.3)
ST_Slope = st.selectbox("ST Slope", options=["Up", "Flat", "Down"])

# -----------------------
# Predict button
# -----------------------
if st.button("Predict Heart Disease"):
    # Prepare data for API
    input_data = {
        "Age": Age,
        "Sex": Sex,
        "ChestPainType": ChestPainType,
        "RestingBP": RestingBP,
        "Cholesterol": Cholesterol,
        "FastingBS": FastingBS,
        "RestingECG": RestingECG,
        "MaxHR": MaxHR,
        "ExerciseAngina": ExerciseAngina,
        "Oldpeak": Oldpeak,
        "ST_Slope": ST_Slope
    }

    try:
        # Call FastAPI endpoint
        response = requests.post("http://127.0.0.1:8000/predict", json=input_data)
        result = response.json()

        st.success(f"Prediction: {'Heart Disease' if result['HeartDiseasePrediction']==1 else 'No Heart Disease'}")
        st.info(f"Probability: {result['Probability']*100:.2f}%")
    except Exception as e:
        st.error(f"Error: {e}")
