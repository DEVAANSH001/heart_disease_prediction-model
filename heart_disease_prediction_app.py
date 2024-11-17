import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier

# Add custom CSS to set the background image from the URL
st.markdown("""
    <style>
    .stApp {
        background-image: url('https://media.licdn.com/dms/image/v2/D5612AQHe7vRE3UNj0w/article-cover_image-shrink_720_1280/article-cover_image-shrink_720_1280/0/1693010842632?e=1737590400&v=beta&t=jSCw4mXvdaAAEveF6_zfKmzMTkdaXtdzs-0M99poO9g');
        background-size: cover;
        background-position: center;
    }
    </style>
    """, unsafe_allow_html=True)

# Load the trained model
model_path = "model.pkl"
with open(model_path, 'rb') as f:
    model = pickle.load(f)

# Define the prediction function
def predict_heart_disease(data):
    # Convert inputs to DataFrame
    input_df = pd.DataFrame([data])
    probability = model.predict_proba(input_df)[:, 1][0]
    ans = "Heart Disease" if probability > 0.5 else "No Heart Disease"
    return ans, probability

# Streamlit UI
st.title("Heart Disease Prediction")

# Collecting user input
age = st.number_input("Age", min_value=1, max_value=120, value=50)
sex = st.selectbox("Sex", options=[0, 1], format_func=lambda x: "Male" if x == 1 else "Female")
cp = st.selectbox("Chest Pain Type", options=[0, 1, 2, 3], format_func=lambda x: ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"][x])
trestbps = st.number_input("Resting Blood Pressure in mm Hg (Ideal range: 94-200)", min_value=80, max_value=200, value=120)
chol = st.number_input("Serum Cholesterol in mg/dL (Ideal range: 126-564)", min_value=100, max_value=600, value=200)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", options=[0, 1], format_func=lambda x: "True" if x == 1 else "False")
restecg = st.selectbox("Resting Electrocardiograph Results", options=[0, 1, 2], format_func=lambda x: ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"][x])
thalach = st.number_input("Maximum Heart Rate Achieved (Ideal range: 71-202)", min_value=50, max_value=220, value=150)
exang = st.selectbox("Exercise Induced Angina", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
oldpeak = st.number_input("ST Depression induced by exercise relative to rest (Ideal range: 0.0-6.2)", min_value=0.0, max_value=10.0, value=1.0)
slope = st.selectbox("Slope of the Peak Exercise ST Segment", options=[0, 1, 2], format_func=lambda x: ["Upslope", "Flat", "Downslope"][x])
ca = st.selectbox("Number of Major Vessels Colored by Fluoroscopy", options=[0, 1, 2, 3], format_func=lambda x: str(x))
thal = st.selectbox("Thalassemia Type", options=[1, 2, 3], format_func=lambda x: {1: "Normal", 2: "Fixed Defect", 3: "Reversible Defect"}[x])

# Creating input dictionary
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

# Prediction
if st.button("Predict"):
    result, probability = predict_heart_disease(input_data)
    st.write("Prediction:", result)
    st.write(f"Probability of Heart Disease: {probability:.2f}")

#streamlit run heart_disease_prediction_app.py 