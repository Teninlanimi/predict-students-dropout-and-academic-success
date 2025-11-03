import streamlit as st
import joblib
import numpy as np

# Load the model
@st.cache_resource
def load_model():
    return joblib.load("multiclass_classification_model.pkl")

model = load_model()

# App layout
st.set_page_config(page_title="Predicting Graduate/Dropout/Enrolled", layout="centered")
st.title("🎓 Predict Academic Outcome")
st.markdown("""
This tool helps determine a student's likely academic outcome — Graduate, Dropout, or Enrolled.
""")

# Form for user input
with st.form("Academic Outcome Prediction Form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        Course = st.selectbox(
            "Select Course",
            ["Architecture", "Computer Science", "Physics", "Accounting", "Civil Engineering"]
        )
        Age_at_enrollment = st.number_input(
            "Age at Enrollment", min_value=16, max_value=90, value=18
        )

    with col2:
        Tuition = st.selectbox("Payment Status", ["No", "Yes"])  
        Special_Needs = st.selectbox("Special Needs", ["No", "Yes"])

    with col3:
        Semester_Registration = st.selectbox("1st Semester Registration", ["Enrolled", "No record"])
        Curriculum_Units = st.selectbox("1st Semester Curriculum Units", ["Approved", "Pending"])

    submitted = st.form_submit_button("Predict")

#  Predict section
if submitted:
    # Encode categorical variables as numbers
    course_map = {
        "Architecture": 0,
        "Computer Science": 1,
        "Physics": 2,
        "Accounting": 3,
        "Civil Engineering": 4
    }
    tuition_val = 1 if Tuition == "Yes" else 0
    special_needs_val = 1 if Special_Needs == "Yes" else 0
    reg_val = 1 if Semester_Registration == "Enrolled" else 0
    curriculum_val = 1 if Curriculum_Units == "Approved" else 0

    # Prepare features as numpy array
    features = np.array([[course_map[Course], Age_at_enrollment, tuition_val,
                          special_needs_val, reg_val, curriculum_val]])

    # Make prediction
    prediction = model.predict(features)[0]

    # Map prediction to label (adjust if needed)
    label_map = {0: "Enrolled", 1: "Graduate", 2: "Dropout"}
    predicted_label = label_map.get(prediction, "Unknown")

    st.success(f" Predicted Academic Outcome: **{predicted_label}**")
