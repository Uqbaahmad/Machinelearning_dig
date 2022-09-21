from optparse import Values
from statistics import mode
import streamlit as st
from joblib import load
import numpy as np
import pandas as pd


@st.cache
def load_model():
    return load('model/diabetes_model_v1.jb')

st.set_page_config(
    page_title="Diabetes churn Prediction",
    layout="wide",
    page_icon=""
)    



with st.form('form1', clear_on_submit=True):
    pregency = st.number_input('Pregnancies',min_value=0.0, max_value=9999.0, value= 10.0, step=.5)
    glucose = st.number_input('Glucose', min_value=0.0, max_value=250.0, value=150.0, step=.5)
    bp = st.number_input('BloodPressure ', min_value=0.0, max_value=250.0, value=80.0, step=.5)
    sthick = st.number_input("SkinThickness", min_value=0.0, max_value=9999.0, value=100.0, step=.5)
    insulin = st.number_input("Insulin", min_value=0.0, max_value=9999.0, value=100.0, step=.5)
    bmi = st.number_input("BMI", min_value=0.0, max_value=100.0, value=40.0)
    dpf = st.number_input("DiabetesPedigreeFunction", min_value=0.0, max_value=9999.0, value=100.0, step=.5)
    age = st.number_input("Age", min_value=18.0, max_value=100.0, value=40.0)
    

    btn = st.form_submit_button("Predict Customer Churn Status")

if btn:
    xinput= [{
        'Pregnancies':pregency,
        'Glucose': glucose,
        'BloodPressure':bp,
        'SkinThickness':sthick,
        'Insulin':insulin,
        'BMI':bmi,
        'DiabetesPedigreeFunction': dpf,
        'Age':age
    }]
    xinput = pd.DataFrame(xinput)
    model = load_model()
    pred = model.predict(xinput)
    st.markdown('# have diabetes' if pred[0] == 0 else '# not diabetes')
    