import streamlit as st
import joblib
import numpy as np
import pandas as pd

# ==========================
# Load the complete pipeline and feature list
# ==========================
@st.cache_resource
def load_pipeline():
    model = joblib.load("multiclass_classification_model.pkl")
    feature_columns = [
        'Marital status', 'Course', 'Daytime/evening attendance', 'Nacionality', 'Displaced',
        'Educational special needs', 'Tuition fees up to date', 'Gender', 'Scholarship holder',
        'Age at enrollment', 'Curricular units 1st sem (enrolled)', 'Curricular units 1st sem (approved)',
        'Curricular units 1st sem (grade)', 'Curricular units 2nd sem (enrolled)',
        'Curricular units 2nd sem (approved)', 'Curricular units 2nd sem (grade)',
        'Unemployment rate', 'Inflation rate', 'GDP'
    ]
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
with st.form("Academic Outcome Prediction Form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        Marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
        Course = st.selectbox(
            "Select Course",
            ["Architecture", "Computer Science", "Physics", "Accounting", "Civil Engineering"]
        )
        Attendance = st.selectbox("Daytime/Evening Attendance", ["Daytime", "Evening"])
        Nacionality = st.selectbox("Nacionality", ["Domestic", "International"])
        Displaced = st.selectbox("Displaced", ["No", "Yes"])
        Special_Needs = st.selectbox("Educational Special Needs", ["No", "Yes"])

    with col2:
        Tuition = st.selectbox("Tuition Fees Up to Date", ["No", "Yes"])
        Gender = st.selectbox("Gender", ["Male", "Female"])
        Scholarship = st.selectbox("Scholarship Holder", ["No", "Yes"])
        Age_at_enrollment = st.number_input("Age at Enrollment", min_value=16, max_value=90, value=18)
        Unemployment_rate = st.number_input("Unemployment Rate (%)", min_value=0.0, max_value=50.0, value=10.0)
        Inflation_rate = st.number_input("Inflation Rate (%)", min_value=0.0, max_value=50.0, value=3.0)

    with col3:
        GDP = st.number_input("GDP Growth (%)", min_value=-10.0, max_value=20.0, value=2.5)
        Sem1_Enrolled = st.number_input("Curricular Units 1st Sem (Enrolled)", min_value=0, value=6)
        Sem1_Approved = st.number_input("Curricular Units 1st Sem (Approved)", min_value=0, value=5)
        Sem1_Grade = st.number_input("Curricular Units 1st Sem (Grade)", min_value=0.0, max_value=20.0, value=12.0)
        Sem2_Enrolled = st.number_input("Curricular Units 2nd Sem (Enrolled)", min_value=0, value=6)
        Sem2_Approved = st.number_input("Curricular Units 2nd Sem (Approved)", min_value=0, value=5)
        Sem2_Grade = st.number_input("Curricular Units 2nd Sem (Grade)", min_value=0.0, max_value=20.0, value=12.0)

    submitted = st.form_submit_button("Predict Outcome")

# ==========================
# Prediction Logic
# ==========================
if submitted:
    # Encode categorical variables
    course_map = {
        "Architecture": 0,
        "Computer Science": 1,
        "Physics": 2,
        "Accounting": 3,
        "Civil Engineering": 4
    }

    marital_map = {"Single": 0, "Married": 1, "Divorced": 2}
    attendance_map = {"Daytime": 0, "Evening": 1}
    nationality_map = {"Domestic": 0, "International": 1}

    displaced_val = 1 if Displaced == "Yes" else 0
    special_needs_val = 1 if Special_Needs == "Yes" else 0
    tuition_val = 1 if Tuition == "Yes" else 0
    gender_val = 1 if Gender == "Female" else 0  # Female = 1
    scholarship_val = 1 if Scholarship == "Yes" else 0

    # Arrange data in a DataFrame using the correct column order
    input_data = pd.DataFrame([{
        'Marital status': marital_map[Marital_status],
        'Course': course_map[Course],
        'Daytime/evening attendance': attendance_map[Attendance],
        'Nacionality': nationality_map[Nacionality],
        'Displaced': displaced_val,
        'Educational special needs': special_needs_val,
        'Tuition fees up to date': tuition_val,
        'Gender': gender_val,
        'Scholarship holder': scholarship_val,
        'Age at enrollment': Age_at_enrollment,
        'Curricular units 1st sem (enrolled)': Sem1_Enrolled,
        'Curricular units 1st sem (approved)': Sem1_Approved,
        'Curricular units 1st sem (grade)': Sem1_Grade,
        'Curricular units 2nd sem (enrolled)': Sem2_Enrolled,
        'Curricular units 2nd sem (approved)': Sem2_Approved,
        'Curricular units 2nd sem (grade)': Sem2_Grade,
        'Unemployment rate': Unemployment_rate,
        'Inflation rate': Inflation_rate,
        'GDP': GDP
    }])

    # Reorder columns to match training
    input_data = input_data.reindex(columns=feature_columns)

    # Predict using the pipeline (scaling included)
    prediction = model.predict(input_data)[0]
    probabilities = model.predict_proba(input_data)[0]

    # Map result to label
    label_map = {0: "Enrolled", 1: "Graduate", 2: "Dropout"}
    predicted_label = label_map.get(prediction, "Unknown")

    # Display result
    st.success(f"🎯 Predicted Academic Outcome: **{predicted_label}**")

    # Show confidence probabilities
    st.markdown("### Prediction Probabilities:")
    prob_df = pd.DataFrame({
        "Outcome": ["Enrolled", "Graduate", "Dropout"],
        "Probability": probabilities
    })
    st.bar_chart(prob_df.set_index("Outcome"))
