import joblib
import pandas as pd
import streamlit as st

st.title("VoltSense AI: Battery Health Prototype")
st.caption("Prototype trained on 4 NASA batteries. Not validated for real vehicles.")

bundle = joblib.load("models/voltsense.joblib")

cycle = st.number_input("Discharge cycle number", 1, 300, 50)
v_mean = st.number_input("Mean voltage (V)", value=3.50)
v_min = st.number_input("Min voltage (V)", value=2.70)
t_mean = st.number_input("Mean temperature (C)", value=32.0)
t_max = st.number_input("Max temperature (C)", value=38.0)

if st.button("Predict"):
    X = pd.DataFrame([[cycle, v_mean, v_min, t_mean, t_max]], columns=bundle["features"])
    soh = bundle["model"].predict(X)[0]
    st.metric("Predicted SOH (relative to first cycle)", f"{soh:.1f}%")
