import streamlit as st 
import pandas as pd
import numpy as np 
import pickle
from sklearn.ensemble import RandomForestClassifier 

st.set_page_config(page_title="Heart Disease Prediction", page_icon="❤️", layout="centered")


model_path = "model.pkl"
with open(model_path, 'rb') as f:
    model = pickle.load(f)


st.sidebar.markdown("##  About This App")
st.sidebar.write("""
This app predicts the **risk of heart disease** based on user input using a trained Random Forest Classifier.

 It uses a trained ML model based on clinical parameters.

 Built with `scikit-learn` and `Streamlit`.
""")

st.sidebar.markdown("---")
st.sidebar.info("Made with by Devaansh")

st.markdown("<h1 style='text-align: center; color: white;'>💓 Heart Disease Predictor</h1>", unsafe_allow_html=True)
st.markdown("<h5 style='text-align: center; color: white;'>Check your heart health in seconds</h5>", unsafe_allow_html=True)

with st.container():
    st.markdown("### 📝 Enter Patient Information")

    
    age = st.number_input("Age", min_value=1, max_value=120, value=50)
    sex = st.selectbox("Sex", options=[0, 1], format_func=lambda x: "Male" if x == 1 else "Female")
    cp = st.selectbox("Chest Pain Type", options=[0, 1, 2, 3], format_func=lambda x: ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"][x])
    trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=80, max_value=200, value=120)
    chol = st.number_input("Serum Cholesterol (mg/dL)", min_value=100, max_value=600, value=200)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", options=[0, 1], format_func=lambda x: "True" if x == 1 else "False")
    restecg = st.selectbox("Resting ECG Results", options=[0, 1, 2], format_func=lambda x: ["Normal", "ST-T Abnormality", "Left Ventricular Hypertrophy"][x])
    thalach = st.number_input("Max Heart Rate Achieved", min_value=50, max_value=220, value=150)
    exang = st.selectbox("Exercise Induced Angina", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    oldpeak = st.number_input("ST Depression (Oldpeak)", min_value=0.0, max_value=10.0, value=1.0)
    slope = st.selectbox("Slope of ST Segment", options=[0, 1, 2], format_func=lambda x: ["Upsloping", "Flat", "Downsloping"][x])
    ca = st.selectbox("Number of Major Vessels Colored", options=[0, 1, 2, 3])
    thal = st.selectbox("Thalassemia", options=[1, 2, 3], format_func=lambda x: {1: "Normal", 2: "Fixed Defect", 3: "Reversible Defect"}[x])


    input_data = {
        'age': age,
        'sex': sex,
        'cp': cp,
        'trestbps': trestbps,
        'chol': chol,
        'fbs': fbs,
        'restecg': restecg,
        'thalach': thalach,
        'exang': exang,
        'oldpeak': oldpeak,
        'slope': slope,
        'ca': ca,
        'thal': thal
    }

# Prediction function
def predict_heart_disease(data):
    input_df = pd.DataFrame([data])
    probability = model.predict_proba(input_df)[:, 1][0]
    diagnosis = "💔 Heart Disease" if probability > 0.5 else "💚 No Heart Disease"
    return diagnosis, probability

if st.button("🧾 Predict Result"):
    result, prob = predict_heart_disease(input_data)
    
    st.markdown("---")
    st.markdown("### 🩺 Prediction Result:")
    st.success(f"**Diagnosis:** {result}")
    st.info(f"**Prediction Confidence:** {prob:.2%}")
    st.markdown("---")

st.markdown("""
    <hr>
    <div style="text-align: center; color: white;">
        © 2025 Devaansh | Built using Streamlit
    </div>
""", unsafe_allow_html=True)