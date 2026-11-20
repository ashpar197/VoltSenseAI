import numpy as np
import streamlit as st

st.title("VoltSense AI: Remaining Life")
st.caption("Rough estimate from your battery's own history. In testing on 4 lab batteries, errors reached about 45 cycles even with 40 to 60 readings.")

text = st.text_area("Paste past SOH values (%), one per line, oldest first:", height=200)

if st.button("Estimate"):
    try:
        soh = [float(x) for x in text.split() if x.strip()]
    except ValueError:
        st.error("Please enter numbers only, one per line.")
        st.stop()

    n = len(soh)
    if n < 40:
        st.info(f"Not enough information. You gave {n} values, and at least 40 are needed for a reliable estimate.")
    elif soh[-1] < 80:
        st.info("The latest SOH is already below 80%, so the battery is past the 80% mark.")
    else:
        x = np.arange(1, n + 1)
        slope, intercept = np.polyfit(x[-20:], np.array(soh[-20:]), 1)
        if slope >= 0:
            st.info("No clear downward trend in the last 20 values, so no estimate is shown.")
        else:
            at80 = (80 - intercept) / slope
            left = at80 - n
            st.metric("Estimated cycles until 80% SOH", f"about {max(left, 0):.0f}")
            st.write(f"Likely range: {max(left - 45, 0):.0f} to {left + 45:.0f} cycles.")
            st.caption("Based on the trend of your last 20 values. Fade usually slows over time and varies by battery, so treat this as rough. It is not a guarantee.")
