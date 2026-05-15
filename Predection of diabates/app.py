import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Page config
st.set_page_config(
    page_title="Diabetes Risk Predictor",
    page_icon="🩺",
    layout="centered"
)

# Custom CSS
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;600&display=swap');

        html, body, [class*="css"] {
            font-family: 'IBM Plex Sans', sans-serif;
        }

        .main {
            background-color: #f0f4f8;
        }

        .stButton > button {
            background-color: #1a73e8;
            color: white;
            border: none;
            padding: 0.6em 2em;
            border-radius: 8px;
            font-size: 16px;
            font-weight: 600;
            width: 100%;
            transition: background-color 0.3s;
        }

        .stButton > button:hover {
            background-color: #1558b0;
        }

        .result-box {
            padding: 1.5em;
            border-radius: 12px;
            text-align: center;
            font-size: 20px;
            font-weight: 600;
            margin-top: 1em;
        }

        .positive {
            background-color: #fde8e8;
            color: #c0392b;
            border: 2px solid #e74c3c;
        }

        .negative {
            background-color: #e8f8f0;
            color: #1e8449;
            border: 2px solid #27ae60;
        }
    </style>
""", unsafe_allow_html=True)

# Load model and scaler
@st.cache_resource
def load_model():
    model = joblib.load('best_model.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

try:
    model, scaler = load_model()
except:
    st.error("⚠️ Could not load model files. Make sure 'best_model.pkl' and 'scaler.pkl' are in the same directory.")
    st.stop()

# Header
st.markdown("## 🩺 Diabetes Risk Predictor")
st.markdown("Enter the patient's clinical data below to predict diabetes risk.")
st.divider()

# Input form
col1, col2 = st.columns(2)

with col1:
    pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1)
    glucose = st.number_input("Glucose (mg/dL)", min_value=0, max_value=300, value=100)
    blood_pressure = st.number_input("Blood Pressure (mm Hg)", min_value=0, max_value=200, value=70)
    skin_thickness = st.number_input("Skin Thickness (mm)", min_value=0, max_value=100, value=20)

with col2:
    insulin = st.number_input("Insulin (μU/mL)", min_value=0, max_value=900, value=80)
    bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0, format="%.1f")
    dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5, format="%.3f")
    age = st.number_input("Age", min_value=1, max_value=120, value=30)

st.divider()

# Predict button
if st.button("🔍 Predict"):
    input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness,
                            insulin, bmi, dpf, age]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    if prediction == 1:
        st.markdown(f"""
            <div class="result-box positive">
                ⚠️ High Risk of Diabetes<br>
                <small>Confidence: {probability:.1%}</small>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
            <div class="result-box negative">
                ✅ Low Risk of Diabetes<br>
                <small>Confidence: {1 - probability:.1%}</small>
            </div>
        """, unsafe_allow_html=True)

    st.divider()
    st.markdown("#### 📊 Input Summary")
    summary = pd.DataFrame({
        'Feature': ['Pregnancies', 'Glucose', 'Blood Pressure', 'Skin Thickness',
                    'Insulin', 'BMI', 'Diabetes Pedigree Function', 'Age'],
        'Value': [pregnancies, glucose, blood_pressure, skin_thickness,
                  insulin, bmi, dpf, age]
    })
    st.dataframe(summary.set_index('Feature'), use_container_width=True)

# Footer
st.markdown("---")
st.markdown("<small>Built with love by Eng. Amr Samir | Random Forest Model | Pima Indians Diabetes Dataset</small>", unsafe_allow_html=True)
