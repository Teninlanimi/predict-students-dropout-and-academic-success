import streamlit as st
import joblib

import numpy as np

#Load the model
@st.cache_resource
def load_model():
    return joblib.load("multiclass_classification_model.pkl")

model = load_model()

#App layout
st.set_page_config(page_title="Predicting Graduate/Dropout/Enrolled", layout="centered")
st.title("Predict Academic Outcome")
st.markdown("""
This tool helps to determine how an individuals academic outcome would be 
""")

# Form for user input
with st.form("Academic Outcome Predicition Form"):
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
        Tuition= st.selectbox("Payment Status", ["No", "Yes"])  
        Special_Needs= st.selectbox("Special Needs", ["No", "Yes"])
        
    with col3:
        Semster_Registeration = st.selectbox("1st Semester", ["Enrolled", "No record"])
        Curriculum_Units = st.selectbox("1st Semester", ["Approved", "Pending"])
        Semster_Registeration = st.selectbox("1st Semester", ["Enrolled", "No record"])
        Curriculum_Units = st.selectbox("2nd Semester", ["Approved", "Pending"])                              

    Acamedic_Status = st.selectbox("Academic Status", ["Graduate", "Dropout", "Enrolled"])

    submitted = st.form_submit_button("Predict")
                          
 #Predict
if submitted:
    Tuition = 1 if is_paid == "Yes" else 0
    SpecialNeeds = 1 if SpecialNeeds == "Yes" else 0
    Status_val = {"Enrolled": 0, "Graduate": 1, "Dropout": 2}[status]

    features = np.array([[Course, Age, Tuition, Special_Needs, Semster_Registeration, Curriculum_Units,  Acamedic_Status]])
    prediction = model.predicct(features)[0]

    st.success(f" Predicted Academic Outcome:{prediction:}")
    