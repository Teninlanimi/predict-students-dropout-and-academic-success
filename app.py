import streamlit as st
import joblib
import numpy as np
import pandas as pd

# ==========================
# Load the complete pipeline and feature list
# ==========================
@st.cache_resource
def load_pipeline():
    # All files are in the same folder, so use the direct filename
    model = joblib.load("multiclass_classification_model.pkl")
    # Load feature list from file to ensure exact match
    with open('final_feature_list.txt', 'r') as f:
        feature_columns = [line.strip() for line in f if line.strip()]
    return model, feature_columns

model, feature_columns = load_pipeline()

# ==========================
# Streamlit Layout
# ==========================
st.set_page_config(page_title=" Predict Academic Outcome", layout="centered")
st.title(" Predict Academic Outcome")
st.markdown("""
This app predicts whether a student is likely to **Graduate**, **Dropout**, or **Remain Enrolled** 
based on key academic and personal information.
""")

# ==========================
# User Input Form
# ==========================

# Dynamically generate input fields matching training data encoding
with st.form("Academic Outcome Prediction Form"):
    user_inputs = {}
    for feature in feature_columns:
        feature_lower = feature.lower()
        # Match input types to training data encoding (numerical values)
        if feature_lower == "marital status":
            marital_options = {"Single": 1, "Married": 2, "Divorced": 3, "Widower": 4, "Divorced": 5, "Facto union": 6}
            marital_choice = st.selectbox(feature, list(marital_options.keys()))
            user_inputs[feature] = marital_options[marital_choice]
        elif feature_lower == "course":
            # Use number input for course ID (training data has course IDs like 171, 9254, etc.)
            user_inputs[feature] = st.number_input(feature + " (Course ID)", min_value=1, value=9254, step=1)
        elif feature_lower == "daytime/evening attendance":
            attendance_options = {"Daytime": 1, "Evening": 0}
            attendance_choice = st.selectbox(feature, list(attendance_options.keys()))
            user_inputs[feature] = attendance_options[attendance_choice]
        elif feature_lower == "nacionality":
            # Nationality is encoded numerically in training data
            user_inputs[feature] = st.number_input(feature + " (1=Nigerian, other=code)", min_value=1, value=1, step=1)
        elif feature_lower in ["displaced", "educational special needs", "tuition fees up to date", "scholarship holder"]:
            binary_options = {"No": 0, "Yes": 1}
            binary_choice = st.selectbox(feature, list(binary_options.keys()))
            user_inputs[feature] = binary_options[binary_choice]
        elif feature_lower == "gender":
            gender_options = {"Male": 0, "Female": 1}
            gender_choice = st.selectbox(feature, list(gender_options.keys()))
            user_inputs[feature] = gender_options[gender_choice]
        elif "enrolled" in feature_lower or "approved" in feature_lower:
            user_inputs[feature] = st.number_input(feature, min_value=0, max_value=30, value=0, step=1)
        elif "grade" in feature_lower:
            user_inputs[feature] = st.number_input(feature, min_value=0.0, max_value=20.0, value=10.0, step=0.1)
        elif feature_lower == "age at enrollment":
            user_inputs[feature] = st.number_input(feature, min_value=16, max_value=90, value=20, step=1)
        elif feature_lower == "unemployment rate":
            user_inputs[feature] = st.number_input(feature, min_value=0.0, max_value=50.0, value=10.8, step=0.1)
        elif feature_lower == "inflation rate":
            user_inputs[feature] = st.number_input(feature, min_value=-10.0, max_value=10.0, value=1.4, step=0.1)
        elif feature_lower == "gdp":
            user_inputs[feature] = st.number_input(feature, min_value=-10.0, max_value=10.0, value=1.74, step=0.01)
        else:
            # Default to number input for unknown numeric features
            user_inputs[feature] = st.number_input(feature, value=0.0)
    submitted = st.form_submit_button("Predict Outcome")

# ==========================
# Prediction Logic
# ==========================
if submitted:
    # Data is already in numerical format matching training data
    input_data = pd.DataFrame([user_inputs])

    prediction = model.predict(input_data)[0]
    probabilities = model.predict_proba(input_data)[0]

    # Map prediction to label based on training encoding
    # Training: Graduate=2, Dropout=1, Enrolled=0
    label_map = {0: "Enrolled", 1: "Dropout", 2: "Graduate"}
    predicted_label = label_map.get(prediction, str(prediction))
    
    # Get class names for display
    if hasattr(model, 'classes_'):
        class_names = [label_map.get(cls, str(cls)) for cls in model.classes_]
    else:
        class_names = ["Enrolled", "Dropout", "Graduate"]

    st.success(f" Predicted Academic Outcome: **{predicted_label}**")

    st.markdown("### Prediction Probabilities:")
    prob_df = pd.DataFrame({
        "Outcome": class_names,
        "Probability": probabilities
    })
    st.bar_chart(prob_df.set_index("Outcome"))
