import streamlit as st
import pandas as pd
import sklearn
import joblib
import numpy as np

# Load the trained pipeline
try:
    with open('models/logistic_regression_pipeline (1).pkl', 'rb') as f:
        pipeline = joblib.load(f)
except FileNotFoundError:
    st.error("Error: logistic_regression_pipeline.pkl not found. Please make sure the file is in the same directory.")
    st.stop()

# Define feature columns (same order as training)
feature_cols = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure', 'PhoneService',
                'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
                'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
                'Contract', 'PaperlessBilling', 'PaymentMethod', 'MonthlyCharges', 'TotalCharges']

st.title("📊 Customer Churn Prediction App")
st.write("Answer the questions below about the customer. The app will predict whether they are likely to leave (churn) or stay.")

# Input fields with proper encoding
input_data = {}

# Gender
gender = st.radio("Customer Gender", ["Female", "Male"], help="Select whether the customer is male or female.")
input_data['gender'] = 0 if gender == "Female" else 1

# Senior Citizen
senior = st.radio("Is the customer a senior citizen?", ["No", "Yes"], help="Senior citizens are considered age 65 or older.")
input_data['SeniorCitizen'] = 1 if senior == "Yes" else 0

# Partner
partner = st.radio("Does the customer have a partner?", ["No", "Yes"])
input_data['Partner'] = 1 if partner == "Yes" else 0

# Dependents
dependents = st.radio("Does the customer have dependents?", ["No", "Yes"])
input_data['Dependents'] = 1 if dependents == "Yes" else 0

# Tenure
input_data['tenure'] = st.number_input("Months customer stayed with company", min_value=0, max_value=100, value=12)

# Phone Service
phone = st.radio("Does the customer have a phone service?", ["No", "Yes"])
input_data['PhoneService'] = 1 if phone == "Yes" else 0

# Multiple Lines (only if phone service is Yes)
if phone == "Yes":
    mult_lines = st.radio("Does the customer have multiple phone lines?", ["No", "Yes"])
    input_data['MultipleLines'] = 1 if mult_lines == "Yes" else 0
else:
    input_data['MultipleLines'] = 0

# Internet Service
internet = st.selectbox("Type of Internet Service", ["No Internet", "DSL", "Fiber optic"])
input_data['InternetService'] = {"No Internet": 0, "DSL": 1, "Fiber optic": 2}[internet]

# Internet-related services (only if InternetService != No Internet)
if internet != "No Internet":
    sec = st.radio("Does the customer have online security service?", ["No", "Yes"])
    input_data['OnlineSecurity'] = 1 if sec == "Yes" else 0

    backup = st.radio("Does the customer have online backup service?", ["No", "Yes"])
    input_data['OnlineBackup'] = 1 if backup == "Yes" else 0

    device = st.radio("Does the customer have device protection service?", ["No", "Yes"])
    input_data['DeviceProtection'] = 1 if device == "Yes" else 0

    tech = st.radio("Does the customer have technical support?", ["No", "Yes"])
    input_data['TechSupport'] = 1 if tech == "Yes" else 0
else:
    input_data['OnlineSecurity'] = 0
    input_data['OnlineBackup'] = 0
    input_data['DeviceProtection'] = 0
    input_data['TechSupport'] = 0

# Streaming Services (always asked, independent of Internet)
stream_tv = st.radio("Does the customer subscribe to streaming TV?", ["No", "Yes"])
input_data['StreamingTV'] = 1 if stream_tv == "Yes" else 0

stream_movies = st.radio("Does the customer subscribe to streaming movies?", ["No", "Yes"])
input_data['StreamingMovies'] = 1 if stream_movies == "Yes" else 0

# Contract
contract = st.selectbox("Customer's contract type", ["Month-to-month", "One year", "Two year"])
input_data['Contract'] = {"Month-to-month": 0, "One year": 1, "Two year": 2}[contract]

# Paperless Billing
paperless = st.radio("Does the customer use paperless billing?", ["No", "Yes"])
input_data['PaperlessBilling'] = 1 if paperless == "Yes" else 0

# Payment Method
payment = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
input_data['PaymentMethod'] = {"Electronic check": 0, "Mailed check": 1, "Bank transfer (automatic)": 2, "Credit card (automatic)": 3}[payment]

# Monthly Charges
input_data['MonthlyCharges'] = st.number_input("Monthly Charges (in USD)", min_value=0.0, value=50.0)

# Total Charges
input_data['TotalCharges'] = st.number_input("Total Charges (in USD)", min_value=0.0, value=1000.0)

# Create DataFrame
input_df = pd.DataFrame([input_data])
input_df = input_df[feature_cols]

# Prediction
if st.button("🔮 Predict Churn"):
    prediction = pipeline.predict(input_df)

    if prediction[0] == 1:
        st.subheader("  Customer is likely to CHURN (leave).")
    else:
        st.subheader(" Customer is likely to STAY.")
