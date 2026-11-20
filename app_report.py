import joblib
import numpy as np
import pandas as pd
import streamlit as st
from recommend import recommend
from pdf_report import generate_pdf_report

st.set_page_config(page_title="VoltSense AI", layout="wide")
st.title("VoltSense AI: Battery Health Prototype")
st.caption("Prototype trained on 4 NASA batteries. Not validated for real vehicles.")

tab1, tab2, tab3 = st.tabs(["Predict Health", "Fade Curves", "Remaining Life"])

with tab1:
    st.subheader("Predict SOH & Reusability")
    bundle = joblib.load("models/voltsense.joblib")

    c1, c2 = st.columns(2)
    with c1:
        cycle = st.number_input("Discharge cycle number", 1, 300, 50)
        v_mean = st.number_input("Mean voltage (V)", value=3.50)
        v_min = st.number_input("Min voltage (V)", value=2.70)
    with c2:
        t_mean = st.number_input("Mean temperature (°C)", value=32.0)
        t_max = st.number_input("Max temperature (°C)", value=38.0)

    if st.button("Predict SOH", key="predict_btn"):
        X = pd.DataFrame([[cycle, v_mean, v_min, t_mean, t_max]], columns=bundle["features"])
        predicted_soh = float(bundle["model"].predict(X)[0])
        rec = recommend(predicted_soh)

        st.session_state["last_prediction"] = {
            "soh": predicted_soh,
            "recommendation": rec,
            "inputs": {
                "cycle": cycle,
                "v_mean": v_mean,
                "v_min": v_min,
                "t_mean": t_mean,
                "t_max": t_max
            }
        }

    # Show results if a prediction has been made
    if "last_prediction" in st.session_state:
        pred_data = st.session_state["last_prediction"]
        soh = pred_data["soh"]
        r = pred_data["recommendation"]
        inputs = pred_data["inputs"]

        st.divider()
        st.subheader("Results")
        res_c1, res_c2 = st.columns(2)
        res_c1.metric("Battery Health (SOH)", f"{soh:.1f}%")
        res_c2.metric("Condition", r["condition"])
        st.progress(min(max(int(soh), 0), 100))

        # PDF Download Button
        pdf_data = generate_pdf_report(inputs, soh, r)
        st.download_button(
            label="Download Battery Health Report (PDF)",
            data=pdf_data,
            file_name=f"voltsense_report_cycle_{inputs['cycle']}.pdf",
            mime="application/pdf",
            key="pdf_download_btn"
        )

        st.subheader("Remaining Life")
        if soh < 80:
            st.info("The predicted SOH is already below 80%, so the battery is past the 80% mark.")
        else:
            st.info("Remaining cycles: not enough information. A reliable estimate needs a history of at least 40 past readings, and one reading cannot show how fast this battery fades.")

        st.subheader("Recommended Uses")
        rec_a, rec_b = st.columns(2)
        with rec_a:
            st.success("May suit:\n\n" + "\n".join("- " + x for x in r["suitable"]))
        with rec_b:
            st.warning("May not suit:\n\n" + "\n".join("- " + x for x in r["unsuitable"]))

        st.write("**Why:** " + r["why"])
        st.caption(r["caveat"])
        st.caption("Typical SOH error is about 3 points in testing on 4 lab batteries. The 80%, 70% and 50% cutoffs are rules of thumb, not validated limits.")

with tab2:
    st.subheader("Fade Curves")
    st.caption("SOH over time for the 4 NASA lab batteries. Each fades at a different speed, and the rate slows over time. The 80% mark is a rule of thumb.")
    df = pd.read_csv("data/processed/battery_features.csv")
    df["soh"] = df.groupby("battery_id")["capacity_ah"].transform(lambda s: s / s.iloc[0] * 100)
    wide = df.pivot(index="cycle", columns="battery_id", values="soh")
    wide["80% line"] = 80
    st.line_chart(wide)

with tab3:
    st.subheader("Estimate Remaining Life")
    st.caption("Rough estimate from your battery's own history. In testing on 4 lab batteries, errors reached about 45 cycles even with 40 to 60 readings.")
    text = st.text_area("Paste past SOH values (%), one per line, oldest first:", height=200, key="soh_history")

    if st.button("Estimate Remaining Life", key="estimate_btn"):
        try:
            soh_vals = [float(x) for x in text.split() if x.strip()]
        except ValueError:
            st.error("Please enter numbers only, one per line.")
            st.stop()

        n = len(soh_vals)
        if n < 40:
            st.info(f"Not enough information. You gave {n} values, and at least 40 are needed for a reliable estimate.")
        elif soh_vals[-1] < 80:
            st.info("The latest SOH is already below 80%, so the battery is past the 80% mark.")
        else:
            x = np.arange(1, n + 1)
            slope, intercept = np.polyfit(x[-20:], np.array(soh_vals[-20:]), 1)
            if slope >= 0:
                st.info("No clear downward trend in the last 20 values, so no estimate is shown.")
            else:
                at80 = (80 - intercept) / slope
                left = at80 - n
                st.metric("Estimated cycles until 80% SOH", f"about {max(left, 0):.0f}")
                st.write(f"Likely range: {max(left - 45, 0):.0f} to {left + 45:.0f} cycles.")
                st.caption("Based on the trend of your last 20 values. Fade usually slows over time and varies by battery, so treat this as rough. It is not a guarantee.")
