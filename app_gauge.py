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
    soh = float(bundle["model"].predict(X)[0])

    st.metric("Predicted SOH (relative to first cycle)", f"{soh:.1f}%")
    st.progress(min(max(int(soh), 0), 100))

    if soh >= 80:
        st.success("Healthy (80% or above)")
    elif soh >= 70:
        st.warning("Aging (70% to 80%)")
    else:
        st.error("Degraded (below 70%)")

    st.caption("Typical error is about 3 SOH points in testing, so treat the status as approximate. The 80% and 70% cutoffs are common rules of thumb, not validated limits.")
