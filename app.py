import streamlit as st
import joblib
import numpy as np

best_model = joblib.load('model.joblib')
scaler = joblib.load('scaler.joblib')

st.title('Diabetes Progression Tracker')
st.write('This app helps you explore your diabetes risk based on the health information you provide.Simply enter the requested values in the fields below.Once you’ve completed all the fields, click Predict to see the result.This prediction is for educational purposes and does not replace medical advice.') 

# Example input — repeat st.number_input for each feature in your dataset
age = st.number_input('Age') 
sex_choice = st.selectbox('Sex', ['Female', 'Male'])

sex = 1 if sex_choice == 'Female' else 2

bmi = st.number_input('Body Mass Index(BMI)')
bp  = st.number_input('Blood Pressure')
s1  = st.number_input('Total Cholesterol')
s2  = st.number_input('Low Density Level Cholesterol(LDL)')
s3  = st.number_input('High Density Level Cholesterol(HDL)')
s4  = st.number_input('Cholesterol/HDL')
s5  = st.number_input('Serum Triglycerides Level')
s6  = st.number_input('Blood Glucose Level')

if st.button('Predict'):
    input_data = np.array([[age, sex, bmi, bp, s1, s2, s3, s4, s5, s6]])
    scaled_input = scaler.transform(input_data)
    result = best_model.predict(scaled_input)
    prediction = float(result[0])

# Adding visuals to emphasize risk
    if prediction < 100:
        st.success(f"Prediction Score: {prediction:.2f} — Low Disease Progression Risk")
    elif 100 <= prediction < 200:
        st.warning(f"Prediction Score: {prediction:.2f} — Moderate Disease Progression Risk")
    else:
        st.error(f"Prediction Score: {prediction:.2f} — High Disease Progression Risk!")

st.caption("This is a learning demo, not a real medical tool.")
