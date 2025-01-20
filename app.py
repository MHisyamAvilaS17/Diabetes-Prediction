import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import pickle


@st.cache_resource
def load_best_model():
    return load_model('best_model_wide_deep.h5')

@st.cache_resource
def load_scaler():
    scaler = load('Diabetes.pkl')

def predict_diabetes(data, model, scaler):
    """
    Fungsi untuk melakukan prediksi diabetes pada data baru.
    """
    # Standarisasi data jika scaler digunakan
    data_scaled = scaler.transform([data])
    prob = model.predict(data_scaled)[0][0]
    return prob > 0.5, prob

st.title("Prediksi Diabetes")
st.write("Masukkan data pasien untuk memprediksi apakah pasien menderita diabetes atau tidak.")

Pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, step=1, value=0)
Glucose = st.number_input("Glucose", min_value=0, max_value=300, step=1, value=120)
BloodPressure = st.number_input("Blood Pressure", min_value=0, max_value=200, step=1, value=80)
SkinThickness = st.number_input("Skin Thickness", min_value=0, max_value=100, step=1, value=20)
Insulin = st.number_input("Insulin", min_value=0, max_value=900, step=1, value=85)
BMI = st.number_input("BMI", min_value=0.0, max_value=70.0, step=0.1, value=25.0)
DiabetesPedigreeFunction = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=2.5, step=0.01, value=0.5)
Age = st.number_input("Age", min_value=0, max_value=120, step=1, value=30)

input_data = [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]

if st.button("Prediksi"):
    model = load_best_model()
    scaler = load_scaler()

    predicted_class, probability = predict_diabetes(input_data, model, scaler)
    status = "Diabetes" if predicted_class else "Tidak Diabetes"

    st.write(f"**Hasil Prediksi**: Pasien diprediksi **{status}**")
    st.write(f"**Probabilitas**: {probability:.2%}")
