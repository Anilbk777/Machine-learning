# app.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle

# ---------------------------
# Load saved objects
# ---------------------------
model = pickle.load(open('churn_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))
le_gender = pickle.load(open('le_gender.pkl', 'rb'))
le_churn = pickle.load(open('le_churn.pkl', 'rb'))
ohe = pickle.load(open('ohe.pkl', 'rb'))

# Load feature columns order saved during training
with open("feature_columns.pkl", "rb") as f:
    feature_columns = pickle.load(f)

# ---------------------------
# Streamlit UI
# ---------------------------
st.title("📊 Customer Churn Prediction")

st.header("Enter Customer Details")

# Numerical inputs
age = st.number_input("Age", min_value=18, max_value=100, value=30)
tenure = st.number_input("Tenure (months)", min_value=0, max_value=72, value=12)
monthly_charges = st.number_input("Monthly Charges", min_value=0.0, value=50.0)

# Categorical inputs
gender = st.selectbox("Gender", le_gender.classes_)  # ['Female', 'Male']
contract = st.selectbox("Contract Type", ohe.categories_[0])
internet = st.selectbox("Internet Service", ohe.categories_[1])
support = st.selectbox("Tech Support", ohe.categories_[2])

# ---------------------------
# Encode inputs
# ---------------------------
# Encode gender
gender_encoded = le_gender.transform([gender])[0]

# One-hot encode contract, internet, support
encoded = ohe.transform(pd.DataFrame([[contract, internet, support]],
                                     columns=['ContractType', 'InternetService', 'TechSupport']))

# Combine numerical + encoded categorical features
numerical_features = np.array([[age, tenure, monthly_charges, gender_encoded]])
final_input = np.concatenate([numerical_features, encoded], axis=1)

# Convert to DataFrame with proper column names
final_input_df = pd.DataFrame(final_input, columns=['Age', 'Tenure', 'MonthlyCharges', 'Gender'] + list(ohe.get_feature_names_out()))

# Reorder columns to match training
final_input_df = final_input_df[feature_columns]

# Scale numerical features
numerical_cols = ['Age', 'Tenure', 'MonthlyCharges']
final_input_df[numerical_cols] = scaler.transform(final_input_df[numerical_cols])

# ---------------------------
# Predict button
# ---------------------------
if st.button("Predict Churn"):
    pred_numeric = model.predict(final_input_df)[0]
    pred_label = le_churn.inverse_transform([int(pred_numeric)])[0]  # 'Yes' or 'No'

    st.write(f"### Prediction: **{pred_label}**")
