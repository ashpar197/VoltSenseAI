import pandas as pd
import streamlit as st

st.title("VoltSense AI: Fade Curves")
st.caption("SOH over time for the 4 NASA lab batteries. Each fades at a different speed, and the rate slows over time. The 80% mark is a rule of thumb.")

df = pd.read_csv("data/processed/battery_features.csv")
df["soh"] = df.groupby("battery_id")["capacity_ah"].transform(lambda s: s / s.iloc[0] * 100)

wide = df.pivot(index="cycle", columns="battery_id", values="soh")
wide["80% line"] = 80
st.line_chart(wide)
