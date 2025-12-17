import streamlit as st
import pickle
import numpy as np

# Load the trained model
with open("logistic_regression_model.pkl", "rb") as f:
    model = pickle.load(f)

st.set_page_config(page_title="Diabetes Prediction", page_icon="🩺")

st.title("🩺 Diabetes Prediction App")
st.write("Enter patient details to predict diabetes")

# Input fields
pregnancies = st.number_input("Pregnancies", min_value=0, step=1)
glucose = st.number_input("Glucose Level", min_value=0)
blood_pressure = st.number_input("Blood Pressure", min_value=0)
skin_thickness = st.number_input("Skin Thickness", min_value=0)
insulin = st.number_input("Insulin Level", min_value=0)
bmi = st.number_input("BMI", min_value=0.0)
diabetes_pedigree = st.number_input("Diabetes Pedigree Function", min_value=0.0)
age = st.number_input("Age", min_value=0, step=1)

# Prediction button
if st.button("Predict Diabetes"):
    input_data = np.array([[pregnancies,
                             glucose,
                             blood_pressure,
                             skin_thickness,
                             insulin,
                             bmi,
                             diabetes_pedigree,
                             age]])

    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.error("⚠️ The person is likely to have Diabetes")
    else:
        st.success("✅ The person is NOT likely to have Diabetes")

