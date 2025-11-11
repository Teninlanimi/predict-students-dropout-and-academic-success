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
st.set_page_config(page_title="🎓 Predict Academic Outcome", layout="centered")
st.title("🎓 Predict Academic Outcome")
st.markdown("""
This app predicts whether a student is likely to **Graduate**, **Dropout**, or **Remain Enrolled** 
based on key academic and personal information.
""")

# ==========================
# User Input Form
# ==========================

# Dynamically generate input fields for all features in final_feature_list.txt
with st.form("Academic Outcome Prediction Form"):
    user_inputs = {}
    for feature in feature_columns:
        # Simple heuristics for input type
        if feature.lower().startswith("marital"):
            user_inputs[feature] = st.selectbox(feature, ["Single", "Married", "Divorced"])
        elif feature.lower() == "course":
            user_inputs[feature] = st.selectbox(feature, ["Architecture", "Computer Science", "Physics", "Accounting", "Civil Engineering"])
        elif "attendance" in feature.lower():
            user_inputs[feature] = st.selectbox(feature, ["Daytime", "Evening"])
        elif feature.lower() == "nacionality":
            user_inputs[feature] = st.selectbox(feature, ["Domestic", "International"])
        elif feature.lower() in ["displaced", "educational special needs", "tuition fees up to date", "scholarship holder"]:
            user_inputs[feature] = st.selectbox(feature, ["No", "Yes"])
        elif feature.lower() == "gender":
            user_inputs[feature] = st.selectbox(feature, ["Male", "Female"])
        elif "enrolled" in feature.lower() or "approved" in feature.lower() or "credited" in feature.lower() or "evaluations" in feature.lower() or "without evaluations" in feature.lower():
            user_inputs[feature] = st.number_input(feature, min_value=0, value=0)
        elif "grade" in feature.lower():
            user_inputs[feature] = st.number_input(feature, min_value=0.0, max_value=20.0, value=10.0)
        elif feature.lower() == "age at enrollment":
            user_inputs[feature] = st.number_input(feature, min_value=16, max_value=90, value=18)
        elif feature.lower() == "unemployment rate" or feature.lower() == "inflation rate":
            user_inputs[feature] = st.number_input(feature, min_value=0.0, max_value=50.0, value=3.0)
        elif feature.lower() == "gdp":
            user_inputs[feature] = st.number_input(feature, min_value=-10.0, max_value=20.0, value=2.5)
        else:
            # Default to text input for unknown types
            user_inputs[feature] = st.text_input(feature, "")
    submitted = st.form_submit_button("Predict Outcome")

# ==========================
# Prediction Logic
# ==========================
if submitted:
    # Encoding for categorical variables (must match training)
    def encode_value(feature, value):
        if feature.lower().startswith("marital"):
            return {"Single": 0, "Married": 1, "Divorced": 2}.get(value, 0)
        elif feature.lower() == "course":
            return {"Architecture": 0, "Computer Science": 1, "Physics": 2, "Accounting": 3, "Civil Engineering": 4}.get(value, 0)
        elif "attendance" in feature.lower():
            return {"Daytime": 0, "Evening": 1}.get(value, 0)
        elif feature.lower() == "nacionality":
            return {"Domestic": 0, "International": 1}.get(value, 0)
        elif feature.lower() in ["displaced", "educational special needs", "tuition fees up to date", "scholarship holder"]:
            return 1 if value == "Yes" else 0
        elif feature.lower() == "gender":
            return 1 if value == "Female" else 0
        else:
            # Try to convert to float, fallback to 0
            try:
                return float(value)
            except Exception:
                return 0

    input_dict = {feature: encode_value(feature, user_inputs[feature]) for feature in feature_columns}
    input_data = pd.DataFrame([input_dict])

    prediction = model.predict(input_data)[0]
    probabilities = model.predict_proba(input_data)[0]

    # Try to get class names from model if available
    if hasattr(model, 'classes_'):
        class_names = list(model.classes_)
    else:
        class_names = ["Enrolled", "Graduate", "Dropout"]

    # Map prediction to label if possible
    label_map = {0: "Enrolled", 1: "Graduate", 2: "Dropout"}
    predicted_label = label_map.get(prediction, str(prediction))

    st.success(f"🎯 Predicted Academic Outcome: **{predicted_label}**")

    st.markdown("### Prediction Probabilities:")
    prob_df = pd.DataFrame({
        "Outcome": class_names,
        "Probability": probabilities
    })
    st.bar_chart(prob_df.set_index("Outcome"))
