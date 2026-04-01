import streamlit as st
import joblib
import pandas as pd
import numpy as np

#load the model and label encoder
model = joblib.load("employee_attrition.pkl")
label_encoder = joblib.load("label_encoder.pkl")
feature_columns = joblib.load("feature_columns.pkl")

#Set up the streamlit app
st.title("Employee Attrition Prediction")
st.markdown("Enter the employee details to predict whether they are" "Likely to leave the company.")

#Create input fields for the features
st.sidebar.header("Employee Details")

def user_input_features():
    inputs={}
    inputs['Age'] =st.sidebar.number_input("Age", min_value=18, max_value=65, value=30)
    inputs['MonthlyIncome']=st.sidebar.number_input("Monthly Income", min_value=1000, max_value=20000,value=5000)
    inputs['JobSatisfaction']=st.sidebar.selectbox("Job Satisfaction", [1, 2, 3, 4])
    inputs['OverTime']=st.sidebar.selectbox("Over Time", ["Yes", "No"])
    inputs['DistanceFromHome']=st.sidebar.number_input("Distance From Home", min_value=0, max_value=50, value=10)

    data = {}
    for feat in feature_columns:
        if feat in inputs:
            data[feat]=inputs[feat]
        else:
            data[feat]=0  #default value for missing features
    return pd.DataFrame(data, index=[0])
        
input_df = user_input_features()

#encoder the 'overtime' features
input_df['OverTime'] = label_encoder['OverTime'].transform(input_df['OverTime'])
#make predictions
if st.button("Prediction Attrition"):
    prediction=model.predict(input_df)
    prediction_proba=model.predict_proba(input_df)

    st.subheader("Prediction")
    if prediction[0] == 1:
        st.error("The employee is likely to leave the company.")
    else:
        st.success("The employee is likely to stay with the company.")

    st.subheader("Prediction Probability")
    st.write(f"Probability of leaving: {prediction_proba[0][1]:.2f}")