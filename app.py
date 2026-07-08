import streamlit as st
import pandas as pd
import joblib

# Load the trained model and scaler
model = joblib.load('logistic_regression_churn_model.pkl')
scaler = joblib.load('scaler.pkl')
model_columns = joblib.load('model_columns.pkl')

st.title("Customer Churn Prediction")
st.write("Enter customer details to predict churn probability.")
# Create input fields for user to enter customer details
tenure = st.number_input("Tenure (in months)", 0,72, 12)
monthly_charges = st.number_input("Monthly Charges", 0.0, 150.0, 70.0)
total_charges = st.number_input("Total Charges", 0.0, 9000.0, 1000.0)
contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
payment_method = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
partner = st.selectbox("Partner", ["No", "Yes"])

if st.button("Predict Churn"):
    input_dict = {col: 0 for col in model_columns}
    
    input_dict['tenure'] = tenure
    input_dict['MonthlyCharges'] = monthly_charges
    input_dict['TotalCharges'] = total_charges
    input_dict['SeniorCitizen'] = 1 if senior_citizen == "Yes" else 0
    
    if f'Contract_{contract}' in input_dict:
        input_dict[f'Contract_{contract}'] = 1
    if f'InternetService_{internet_service}' in input_dict:
        input_dict[f'InternetService_{internet_service}'] = 1
    if f'PaymentMethod_{payment_method}' in input_dict:
        input_dict[f'PaymentMethod_{payment_method}'] = 1
    if partner == "Yes" and 'Partner_Yes' in input_dict:
        input_dict['Partner_Yes'] = 1

    input_df = pd.DataFrame([input_dict])
    input_scaled = scaler.transform(input_df)
    
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]
    
    if prediction == 1:
        st.error(f" Customer likely to CHURN (Probability: {probability:.2%})")
    else:
        st.success(f"Customer likely to STAY (Probability of churn: {probability:.2%})")