# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 11:08:33 2024

@author: EVTHAG
"""

import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder, StandardScaler

with open('clf .pkl', 'rb') as f:
    model = pickle.load(f)
    
with open('education_label_encoder.pkl', 'rb') as f:
    label_encoders = pickle.load(f)
    
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

trained_features = [
    'age', 'duration', 'campaign', 'pdays', 'previous', 'emp.var.rate',
       'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed',
       'education_encoded', 'job_blue-collar', 'job_entrepreneur',
       'job_housemaid', 'job_management', 'job_retired', 'job_self-employed',
       'job_services', 'job_student', 'job_technician', 'job_unemployed',
       'marital_married', 'marital_single', 'housing_yes', 'loan_yes',
       'contact_telephone', 'month_aug', 'month_dec', 'month_jul', 'month_jun',
       'month_mar', 'month_may', 'month_nov', 'month_oct', 'month_sep',
       'day_of_week_mon', 'day_of_week_thu', 'day_of_week_tue',
       'day_of_week_wed', 'poutcome_nonexistent', 'poutcome_success'
    ]

# Preprocessing function
def preprocess_input(input_data):
    """
    Preprocess user inputs for the model prediction.
    - Label encodes specified columns.
    - One-hot encodes categorical features.
    - Scales numerical features.

    Args:
        input_data (pd.DataFrame): Input data as a DataFrame.

    Returns:
        pd.DataFrame: Processed input ready for prediction.
    """
    # Assuming label_encoders is a single LabelEncoder for 'education'
    input_data["education_encoded"] = label_encoders.transform(input_data["education"])
    input_data.drop("education", axis=1, inplace=True)  # Drop original column
    
    # One-hot encode categorical columns
    categorical_cols = ["job", "marital", "housing", "loan", "contact", "month", "day_of_week"]
    input_data = pd.get_dummies(input_data, columns=categorical_cols, drop_first=True)
    
    # Add missing columns with default values (e.g., 0)
    for col in trained_features:
        if col not in input_data.columns:
            input_data[col] = 0
            
 
# Reorder columns to match the order used during training
    input_data = input_data[trained_features]
    
# Scale the input data
    input_data = scaler.transform(input_data)

    return input_data

# Define the Streamlit interface
st.title("Bank Marketing Prediction App")
st.write("This app predicts the likelihood of a successful marketing campaign outcome.")

# Input fields (replace with the actual features used in your model)
age = st.number_input("Age", min_value=18, max_value=100, value=30, step=1)
job = st.selectbox("Job Type", ["admin.", "blue-collar", "entrepreneur", "housemaid", "services", "technician", "unemployed"])
marital = st.selectbox("Marital Status", ["single", "married", "divorced"])
education = st.selectbox("Education Level", ["basic.4y", "basic.6y", "high.school", "university.degree"])
housing = st.selectbox("Housing Loan", ["yes", "no"])
loan = st.selectbox("Personal Loan", ["yes", "no"])
contact = st.selectbox("Contact Type", ["telephone", "cellular"])
month = st.selectbox("Month", ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"])
day_of_week = st.selectbox("Day of the Week", ["mon", "tue", "wed", "thu", "fri"])
duration = st.number_input("Call Duration (seconds)", min_value=0, max_value=5000, value=100)
campaign = st.number_input("Number of Contacts During Campaign", min_value=1, max_value=50, value=1)

# Collect inputs into a DataFrame
input_data = pd.DataFrame({
    "age": [age],
    "job": [job],
    "marital": [marital],
    "education": [education],
    "housing": [housing],
    "loan": [loan],
    "contact": [contact],
    "month": [month],
    "day_of_week": [day_of_week],
    "duration": [duration],
    "campaign": [campaign],
})

# Preprocess the inputs
processed_input = preprocess_input(input_data)

# Make predictions
if st.button("Predict"):
    prediction = model.predict(processed_input)
    st.write(f"The predicted outcome is: **{'Yes' if prediction[0] == 1 else 'No'}**")
