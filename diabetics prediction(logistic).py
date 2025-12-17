import streamlit as st
import pickle
import numpy as np

st.set_page_config(page_title="Diabetes Prediction", page_icon="🩺")

# ================= LOAD MODEL =================
with open("logistic_regression_model.pkl", "rb") as f:
    model = pickle.load(f)

# Try loading scaler (if used during training)
scaler = None
try:
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
except:
    pass

st.title("🩺 Diabetes Prediction App")
st.write("Enter patient details below")

# ================= USER INPUTS =================
pregnancies = st.number_input("Pregnancies", min_value=0, step=1)
glucose = st.number_input("Glucose Level", min_value=0)
blood_pressure = st.number_input("Blood Pressure", min_value=0)
skin_thickness = st.number_input("Skin Thickness", min_value=0)
insulin = st.number_input("Insulin Level", min_value=0)
bmi = st.number_input("BMI", min_value=0.0)
diabetes_pedigree = st.number_input("Diabetes Pedigree Function", min_value=0.0)
age = st.number_input("Age", min_value=0, step=1)

# ================= PREDICTION =================
if st.button("Predict Diabetes"):

    # Create full feature array
    full_input = np.array([[pregnancies,
                             glucose,
                             blood_pressure,
                             skin_thickness,
                             insulin,
                             bmi,
                             diabetes_pedigree,
                             age]])

    # Match model feature count automatically
    expected_features = model.n_features_in_
    input_data = full_input[:, :expected_features]

    # Apply scaler if exists
    if scaler is not None:
        input_data = scaler.transform(input_data)

    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.error("⚠️ The person is likely to have Diabetes")
    else:
        st.success("✅ The person is NOT likely to have Diabetes")

# ================= DEBUG (OPTIONAL) =================
with st.expander("Debug Info"):
    st.write("Model expects features:", model.n_features_in_)
