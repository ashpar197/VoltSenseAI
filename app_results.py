import joblib
import pandas as pd
import streamlit as st
from recommend import recommend

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
    r = recommend(soh)

    st.subheader("Results")
    c1, c2 = st.columns(2)
    c1.metric("Battery health (SOH)", f"{soh:.1f}%")
    c2.metric("Condition", r["condition"])
    st.progress(min(max(int(soh), 0), 100))

    st.subheader("Remaining life")
    if soh < 80:
        st.info("The predicted SOH is already below 80%, so the battery is past the 80% mark.")
    else:
        st.info("Remaining cycles: not enough information. A reliable estimate needs a history of at least 40 past readings, and one reading cannot show how fast this battery fades.")

    st.subheader("Recommended uses")
    a, b = st.columns(2)
    with a:
        st.success("May suit:\n\n" + "\n".join("- " + x for x in r["suitable"]))
    with b:
        st.warning("May not suit:\n\n" + "\n".join("- " + x for x in r["unsuitable"]))

    st.write("**Why:** " + r["why"])
    st.caption(r["caveat"])
    st.caption("Typical SOH error is about 3 points in testing on 4 lab batteries. The 80%, 70% and 50% cutoffs are rules of thumb, not validated limits.")
