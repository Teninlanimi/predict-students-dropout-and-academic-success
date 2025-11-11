import streamlit as st
import joblib
import numpy as np

# Load the model
@st.cache_resource
def load_model():
    return joblib.load("multiclass_classification_model.pkl")

model = load_model()

# App layout
st.set_page_config(page_title="🎓 Predict Academic Outcome", layout="centered")
st.title("🎓 Predict Academic Outcome")
st.markdown("""
This app predicts whether a student is likely to **Graduate**, **Dropout**, or **Remain Enrolled** 
based on key academic and personal information.
""")

# Form for user input
with st.form("Academic Outcome Prediction Form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        Course = st.selectbox(
            "Select Course",
            ["Architecture", "Computer Science", "Physics", "Accounting", "Civil Engineering"]
        )
        Displaced = st.selectbox("Displaced", ["No", "Yes"])
        Special_Needs = st.selectbox("Educational Special Needs", ["No", "Yes"])
        Tuition = st.selectbox("Tuition Fees Up to Date", ["No", "Yes"])
        Gender = st.selectbox("Gender", ["Male", "Female"])
    
    with col2:
        Scholarship = st.selectbox("Scholarship Holder", ["No", "Yes"])
        Age_at_enrollment = st.number_input("Age at Enrollment", min_value=16, max_value=90, value=18)
        Sem1_Enrolled = st.number_input("Curricular Units 1st Sem (Enrolled)", min_value=0, value=6)
        Sem1_Approved = st.number_input("Curricular Units 1st Sem (Approved)", min_value=0, value=5)
        Sem1_Grade = st.number_input("Curricular Units 1st Sem (Grade)", min_value=0.0, max_value=20.0, value=12.0)
    
    with col3:
        Sem2_Enrolled = st.number_input("Curricular Units 2nd Sem (Enrolled)", min_value=0, value=6)
        Sem2_Approved = st.number_input("Curricular Units 2nd Sem (Approved)", min_value=0, value=5)
        Sem2_Grade = st.number_input("Curricular Units 2nd Sem (Grade)", min_value=0.0, max_value=20.0, value=12.0)

    submitted = st.form_submit_button("Predict Outcome")

# Prediction section
if submitted:
    # Encode categorical variables
    course_map = {
        "Architecture": 0,
        "Computer Science": 1,
        "Physics": 2,
        "Accounting": 3,
        "Civil Engineering": 4
    }

    displaced_val = 1 if Displaced == "Yes" else 0
    special_needs_val = 1 if Special_Needs == "Yes" else 0
    tuition_val = 1 if Tuition == "Yes" else 0
    gender_val = 1 if Gender == "Female" else 0  # Assuming Female=1, Male=0
    scholarship_val = 1 if Scholarship == "Yes" else 0

    # Arrange features in the same order the model expects
    features = np.array([[
        course_map[Course], displaced_val, special_needs_val, tuition_val,
        gender_val, scholarship_val, Age_at_enrollment,
        Sem1_Enrolled, Sem1_Approved, Sem1_Grade,
        Sem2_Enrolled, Sem2_Approved, Sem2_Grade
    ]])

    # Make prediction
    prediction = model.predict(features)[0]

    # Map prediction to label
    label_map = {0: "Enrolled", 1: "Graduate", 2: "Dropout"}
    predicted_label = label_map.get(prediction, "Unknown")

    # Display result
    st.success(f"🎯 Predicted Academic Outcome: **{predicted_label}**")
