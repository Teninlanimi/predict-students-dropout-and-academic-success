import streamlit as st
import joblib
import numpy as np

# Load the model
@st.cache_resource
def load_model():
    return joblib.load("multiclass_classification_model.pkl")

model = load_model()

# Streamlit App Configuration
st.set_page_config(page_title="🎓 Academic Outcome Prediction", layout="wide")
st.title("🎓 Predict Academic Outcome")
st.markdown("""
This tool predicts whether a student is likely to **Graduate**, **Dropout**, or remain **Enrolled**  
based on academic, personal, and socio-economic features.
""")

# --------------- FEATURE INPUT SECTION ----------------

with st.form("Academic Outcome Form"):
    st.subheader("Student and Academic Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        Application_mode = st.number_input("Application mode", min_value=0)
        Application_order = st.number_input("Application order", min_value=0)
        Course = st.number_input("Course", min_value=0)
        Previous_qualification = st.number_input("Previous qualification", min_value=0)
        Previous_qualification_grade = st.number_input("Previous qualification (grade)", min_value=0.0)

        Nacionality = st.number_input("Nacionality", min_value=0)
        Mother_qualification = st.number_input("Mother's qualification", min_value=0)
        Father_qualification = st.number_input("Father's qualification", min_value=0)
        Mother_occupation = st.number_input("Mother's occupation", min_value=0)
        Father_occupation = st.number_input("Father's occupation", min_value=0)

        Admission_grade = st.number_input("Admission grade", min_value=0.0)
        Displaced = st.selectbox("Displaced", ["No", "Yes"])
        Educational_special_needs = st.selectbox("Educational special needs", ["No", "Yes"])

    with col2:
        Debtor = st.selectbox("Debtor", ["No", "Yes"])
        Tuition_fees_up_to_date = st.selectbox("Tuition fees up to date", ["No", "Yes"])
        Gender = st.selectbox("Gender", ["Male", "Female"])
        Scholarship_holder = st.selectbox("Scholarship holder", ["No", "Yes"])
        Age_at_enrollment = st.number_input("Age at enrollment", min_value=16, max_value=90, value=18)
        International = st.selectbox("International", ["No", "Yes"])

        CU1_credited = st.number_input("Curricular units 1st sem (credited)", min_value=0)
        CU1_enrolled = st.number_input("Curricular units 1st sem (enrolled)", min_value=0)
        CU1_evaluations = st.number_input("Curricular units 1st sem (evaluations)", min_value=0)
        CU1_approved = st.number_input("Curricular units 1st sem (approved)", min_value=0)
        CU1_grade = st.number_input("Curricular units 1st sem (grade)", min_value=0.0)
        CU1_without_eval = st.number_input("Curricular units 1st sem (without evaluations)", min_value=0)

    with col3:
        CU2_credited = st.number_input("Curricular units 2nd sem (credited)", min_value=0)
        CU2_enrolled = st.number_input("Curricular units 2nd sem (enrolled)", min_value=0)
        CU2_evaluations = st.number_input("Curricular units 2nd sem (evaluations)", min_value=0)
        CU2_approved = st.number_input("Curricular units 2nd sem (approved)", min_value=0)
        CU2_grade = st.number_input("Curricular units 2nd sem (grade)", min_value=0.0)
        CU2_without_eval = st.number_input("Curricular units 2nd sem (without evaluations)", min_value=0)

        Unemployment_rate = st.number_input("Unemployment rate", min_value=0.0)
        Inflation_rate = st.number_input("Inflation rate", min_value=0.0)
        GDP = st.number_input("GDP", min_value=0.0)
        Target_binary = st.number_input("Target_binary (if applicable)", min_value=0)

    submitted = st.form_submit_button("Predict Outcome")

# --------------- PREDICTION SECTION ----------------

if submitted:
    # Encode binary categorical fields (0 = No, 1 = Yes)
    def yes_no(x): return 1 if x == "Yes" else 0

    features = np.array([[
        Application_mode,
        Application_order,
        Course,
        Previous_qualification,
        Previous_qualification_grade,
        Nacionality,
        Mother_qualification,
        Father_qualification,
        Mother_occupation,
        Father_occupation,
        Admission_grade,
        yes_no(Displaced),
        yes_no(Educational_special_needs),
        yes_no(Debtor),
        yes_no(Tuition_fees_up_to_date),
        1 if Gender == "Male" else 0,
        yes_no(Scholarship_holder),
        Age_at_enrollment,
        yes_no(International),
        CU1_credited,
        CU1_enrolled,
        CU1_evaluations,
        CU1_approved,
        CU1_grade,
        CU1_without_eval,
        CU2_credited,
        CU2_enrolled,
        CU2_evaluations,
        CU2_approved,
        CU2_grade,
        CU2_without_eval,
        Unemployment_rate,
        Inflation_rate,
        GDP,
        Target_binary
    ]])

    # Predict
    prediction = model.predict(features)[0]

    # Label map
    label_map = {0: "Enrolled", 1: "Graduate", 2: "Dropout"}
    predicted_label = label_map.get(prediction, "Unknown")

    st.success(f"🎯 Predicted Academic Outcome: **{predicted_label}**")
