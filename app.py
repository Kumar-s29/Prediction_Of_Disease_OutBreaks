import os
import pickle  # For loading pre-trained models
import streamlit as st
from streamlit_option_menu import option_menu
import numpy as np

st.set_page_config(page_title="Prediction of Disease Outbreaks", layout="wide", page_icon="🏥")

# Load the pre-trained models
try:
    diabetes_model = pickle.load(open('K:/Prediction_Of_Disease_OutBreaks/Training_Models/diabetes-prediction-model.sav', 'rb'))
    heart_model = pickle.load(open('K:/Prediction_Of_Disease_OutBreaks/Training_Models/heart_model.sav', 'rb'))
    parkinsons_model = pickle.load(open('K:/Prediction_Of_Disease_OutBreaks/Training_Models/parkinsons_model.sav', 'rb'))
except FileNotFoundError:
    st.error("Error: One or more model files are missing. Please check the file paths.")

# Sidebar Navigation
with st.sidebar:
    selected = option_menu("Prediction of Disease Outbreaks",
                           ["Diabetes Prediction", "Heart Disease Prediction", "Parkinsons Prediction"],
                           menu_icon="hospital-fill",
                           icons=["activity", "heart", "person"],
                           default_index=0)

# Diabetes Prediction
if selected == "Diabetes Prediction":
    st.title("Diabetes Prediction")
    col1, col2, col3 = st.columns(3)

    with col1:
        pregnancies = st.text_input("Pregnancies")
    with col2:
        glucose = st.text_input("Glucose Level")
    with col3:
        blood_pressure = st.text_input("Blood Pressure")
    with col1:
        skin_thickness = st.text_input("Skin Thickness")
    with col2:
        insulin = st.text_input("Insulin")
    with col3:
        bmi = st.text_input("BMI")
    with col1:
        diabetes_pedigree = st.text_input("Diabetes Pedigree Function")
    with col2:
        age = st.text_input("Age")

    diab_diagnosis = ""

    if st.button("Diabetes Test Result"):
        try:
            input_data = [float(i) for i in [pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree, age]]
            diab_prediction = diabetes_model.predict([np.array(input_data)])

            if diab_prediction[0] == 1:
                diab_diagnosis = "The person has diabetes."
            else:
                diab_diagnosis = "The person does not have diabetes."
            st.success(diab_diagnosis)
        except ValueError:
            st.error("Please enter valid numeric values.")

# Heart Disease Prediction
if selected == "Heart Disease Prediction":
    st.title("Heart Disease Prediction")
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.text_input("Age")
    with col2:
        sex = st.text_input("Gender (1 = Male, 0 = Female)")
    with col3:
        cp = st.text_input("Chest Pain Type")
    with col1:
        trestbps = st.text_input("Resting Blood Pressure")
    with col2:
        chol = st.text_input("Cholesterol Level")
    with col3:
        fbs = st.text_input("Fasting Blood Sugar")
    with col1:
        restecg = st.text_input("Resting ECG")
    with col2:
        thalach = st.text_input("Max Heart Rate")
    with col3:
        exang = st.text_input("Exercise Induced Angina")
    with col1:
        oldpeak = st.text_input("ST Depression")
    with col2:
        slope = st.text_input("Slope")
    with col3:
        ca = st.text_input("CA")
    with col1:
        thal = st.text_input("Thal")

    heart_diagnosis = ""

    if st.button("Heart Disease Test Result"):
        try:
            input_data = [float(i) for i in [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]]
            heart_prediction = heart_model.predict([np.array(input_data)])

            if heart_prediction[0] == 1:
                heart_diagnosis = "The person has heart disease."
            else:
                heart_diagnosis = "The person does not have heart disease."
            st.success(heart_diagnosis)
        except ValueError:
            st.error("Please enter valid numeric values.")

# Parkinson's Disease Prediction
if selected == "Parkinsons Prediction":
    st.title("Parkinson’s Prediction")
    col1, col2, col3 = st.columns(3)

    features = ["MDVP:Fo(Hz)", "MDVP:Fhi(Hz)", "MDVP:Flo(Hz)", "MDVP:Jitter(%)", "MDVP:Jitter(Abs)", "MDVP:RAP", 
                "MDVP:PPQ", "Jitter:DDP", "MDVP:Shimmer", "MDVP:Shimmer(dB)", "Shimmer:APQ3", "Shimmer:APQ5", 
                "MDVP:APQ", "Shimmer:DDA", "NHR", "HNR", "RPDE", "DFA", "Spread1", "Spread2", "D2", "PPE"]

    input_values = []
    for i, feature in enumerate(features):
        with (col1 if i % 3 == 0 else col2 if i % 3 == 1 else col3):
            input_values.append(st.text_input(feature))

    parkinsons_diagnosis = ""

    if st.button("Parkinson’s Test Result"):
        try:
            input_data = [float(i) for i in input_values]
            parkinsons_prediction = parkinsons_model.predict([np.array(input_data)])

            if parkinsons_prediction[0] == 1:
                parkinsons_diagnosis = "The person has Parkinson’s disease."
            else:
                parkinsons_diagnosis = "The person does not have Parkinson’s disease."
            st.success(parkinsons_diagnosis)
        except ValueError:
            st.error("Please enter valid numeric values.")
