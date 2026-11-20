import io
from datetime import datetime
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "voltsense.joblib"
DATA_PATH = BASE_DIR / "data" / "processed" / "battery_features.csv"

from recommend import recommend
from pdf_report import generate_pdf_report
from database import (
    init_db,
    save_prediction,
    get_all_predictions,
    get_unique_batteries,
    delete_prediction,
    get_battery_summary
)

# Initialize database schema if not present
init_db()

st.set_page_config(page_title="VoltSense AI", layout="wide", page_icon="⚡")
st.title("⚡ VoltSense AI: Battery Health & Prediction History")
st.caption("Prototype trained on 4 NASA laboratory batteries. Experimental decision-support tool.")

# Four Main Navigation Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "Predict Health",
    "Fade Curves",
    "Remaining Life",
    "Prediction History"
])

# ---------------------------------------------------------
# TAB 1: PREDICT HEALTH
# ---------------------------------------------------------
with tab1:
    st.subheader("Predict SOH & Reusability")
    bundle = joblib.load(MODEL_PATH)

    col_meta1, col_meta2 = st.columns(2)
    with col_meta1:
        battery_id = st.text_input("Battery ID / Pack Name", value="Battery-1", help="Enter a custom name or ID to track this battery's history.")
    with col_meta2:
        cycle = st.number_input("Discharge cycle number", 1, 500, 50)

    c1, c2 = st.columns(2)
    with c1:
        v_mean = st.number_input("Mean voltage (V)", value=3.50, format="%.2f")
        v_min = st.number_input("Min voltage (V)", value=2.70, format="%.2f")
    with c2:
        t_mean = st.number_input("Mean temperature (°C)", value=32.0, format="%.1f")
        t_max = st.number_input("Max temperature (°C)", value=38.0, format="%.1f")

    if st.button("Predict SOH", key="predict_btn", type="primary"):
        clean_id = battery_id.strip() if battery_id.strip() else "Battery-1"
        X = pd.DataFrame([[cycle, v_mean, v_min, t_mean, t_max]], columns=bundle["features"])
        predicted_soh = float(bundle["model"].predict(X)[0])
        rec = recommend(predicted_soh)

        # Remaining life note
        if predicted_soh < 80:
            rul_note = "Past 80% SOH mark"
        else:
            rul_note = "Requires ≥40 past cycles"

        st.session_state["last_prediction"] = {
            "battery_id": clean_id,
            "soh": predicted_soh,
            "recommendation": rec,
            "remaining_life": rul_note,
            "inputs": {
                "cycle": int(cycle),
                "v_mean": float(v_mean),
                "v_min": float(v_min),
                "t_mean": float(t_mean),
                "t_max": float(t_max)
            }
        }

    # Display prediction results
    if "last_prediction" in st.session_state:
        pred = st.session_state["last_prediction"]
        soh = pred["soh"]
        r = pred["recommendation"]
        inputs = pred["inputs"]
        cur_bat = pred["battery_id"]

        st.divider()
        st.subheader("Results")
        res_c1, res_c2, res_c3 = st.columns(3)
        res_c1.metric("Tracked Battery", cur_bat)
        res_c2.metric("Battery Health (SOH)", f"{soh:.1f}%")
        res_c3.metric("Condition", r["condition"])
        st.progress(min(max(int(soh), 0), 100))

        # Action Buttons (Save to Database & Download PDF)
        act1, act2 = st.columns(2)
        with act1:
            if st.button("💾 Save to Prediction History", key="save_history_btn"):
                rec_id = save_prediction(
                    battery_id=cur_bat,
                    soh=soh,
                    condition=r["condition"],
                    cycle=inputs["cycle"],
                    v_mean=inputs["v_mean"],
                    v_min=inputs["v_min"],
                    t_mean=inputs["t_mean"],
                    t_max=inputs["t_max"],
                    remaining_life=pred["remaining_life"]
                )
                st.success(f"✅ Prediction #{rec_id} saved for '{cur_bat}'! View it in the Prediction History tab.")

        with act2:
            pdf_bytes = generate_pdf_report(inputs, soh, r)
            st.download_button(
                label="📄 Download Battery Health Report (PDF)",
                data=pdf_bytes,
                file_name=f"voltsense_{cur_bat}_cycle_{inputs['cycle']}.pdf",
                mime="application/pdf",
                key="pdf_download_btn"
            )

        st.subheader("Remaining Life Assessment")
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


# ---------------------------------------------------------
# TAB 2: FADE CURVES (Original NASA Batteries)
# ---------------------------------------------------------
with tab2:
    st.subheader("Fade Curves")
    st.caption("SOH over time for the 4 NASA lab batteries. Each fades at a different speed, and the rate slows over time. The 80% mark is a rule of thumb.")
    df_curves = pd.read_csv(DATA_PATH)
    df_curves["soh"] = df_curves.groupby("battery_id")["capacity_ah"].transform(lambda s: s / s.iloc[0] * 100)
    wide = df_curves.pivot(index="cycle", columns="battery_id", values="soh")
    wide["80% line"] = 80
    st.line_chart(wide)


# ---------------------------------------------------------
# TAB 3: REMAINING LIFE (Polyfit Historical Estimator)
# ---------------------------------------------------------
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


# ---------------------------------------------------------
# TAB 4: PREDICTION HISTORY & COMPARISON
# ---------------------------------------------------------
with tab4:
    st.subheader("Prediction History & Battery Health Comparison")
    
    unique_batteries = get_unique_batteries()
    
    # 1. Summary Dashboard Metrics Cards
    all_df = get_all_predictions()
    total_records = len(all_df)
    total_bats = len(unique_batteries)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Saved Predictions", total_records)
    m2.metric("Batteries Tracked", total_bats)

    # Battery filter controls
    selected_battery = "All Batteries"
    if unique_batteries:
        filter_col1, filter_col2 = st.columns(2)
        with filter_col1:
            selected_battery = st.selectbox(
                "Filter by Battery ID:",
                options=["All Batteries"] + unique_batteries,
                index=0
            )
        with filter_col2:
            search_query = st.text_input("Search Battery Name:", placeholder="Type to filter...")

    summary = get_battery_summary(selected_battery if selected_battery != "All Batteries" else None)

    latest_soh_display = f"{summary['latest_soh']:.1f}%" if summary["latest_soh"] is not None else "N/A"
    delta_display = f"{summary['soh_change']:+.1f}%" if summary["soh_change"] is not None else None
    
    m3.metric("Latest Recorded SOH", latest_soh_display)
    m4.metric("Change (vs Prev)", delta_display if delta_display else "—")

    if total_records == 0:
        st.info("No predictions saved yet. Make a prediction in the 'Predict Health' tab and click 'Save to Prediction History'.")
    else:
        st.divider()
        
        # 2. Comparison & Trends Section
        st.subheader("Battery Degradation Trend & Comparison")

        compare_mode = False
        if total_bats >= 2:
            compare_mode = st.checkbox("Compare two batteries side-by-side")

        if compare_mode and total_bats >= 2:
            cmp_c1, cmp_c2 = st.columns(2)
            with cmp_c1:
                bat_a = st.selectbox("Battery A:", options=unique_batteries, index=0)
            with cmp_c2:
                default_b_idx = 1 if len(unique_batteries) > 1 else 0
                bat_b = st.selectbox("Battery B:", options=unique_batteries, index=default_b_idx)

            df_a = get_all_predictions(battery_id=bat_a).sort_values("cycle")
            df_b = get_all_predictions(battery_id=bat_b).sort_values("cycle")

            combined_cmp = pd.concat([df_a, df_b])
            if not combined_cmp.empty:
                chart = alt.Chart(combined_cmp).mark_line(point=True).encode(
                    x=alt.X("cycle:Q", title="Discharge Cycle Number"),
                    y=alt.Y("soh:Q", title="Predicted SOH (%)", scale=alt.Scale(zero=False)),
                    color=alt.Color("battery_id:N", title="Battery ID"),
                    tooltip=["battery_id", "cycle", "soh", "condition", "timestamp"]
                ).properties(height=350).interactive()
                st.altair_chart(chart, use_container_width=True)
            else:
                st.info("No cycle data available for the chosen batteries.")

        elif selected_battery != "All Batteries":
            bat_df = get_all_predictions(battery_id=selected_battery).sort_values("cycle")
            if not bat_df.empty:
                chart = alt.Chart(bat_df).mark_line(point=True, color="#2563EB").encode(
                    x=alt.X("cycle:Q", title="Discharge Cycle Number"),
                    y=alt.Y("soh:Q", title="Predicted SOH (%)", scale=alt.Scale(zero=False)),
                    tooltip=["cycle", "soh", "condition", "timestamp", "v_mean", "t_mean"]
                ).properties(height=320).interactive()
                st.altair_chart(chart, use_container_width=True)

                # Battery specific stats
                s_col1, s_col2, s_col3 = st.columns(3)
                s_col1.write(f"**Records for {selected_battery}:** {len(bat_df)}")
                s_col2.write(f"**First Record:** {summary.get('earliest_date', 'N/A')}")
                s_col3.write(f"**Latest Record:** {summary.get('latest_date', 'N/A')}")
        else:
            # All batteries overview
            chart = alt.Chart(all_df).mark_line(point=True).encode(
                x=alt.X("cycle:Q", title="Discharge Cycle Number"),
                y=alt.Y("soh:Q", title="Predicted SOH (%)", scale=alt.Scale(zero=False)),
                color=alt.Color("battery_id:N", title="Battery ID"),
                tooltip=["battery_id", "cycle", "soh", "condition", "timestamp"]
            ).properties(height=350).interactive()
            st.altair_chart(chart, use_container_width=True)

        st.caption("ℹ️ **Scientific Note:** Degradation curves only connect readings belonging to the same battery. Small upward bumps reflect normal measurement variation or temperature changes, not battery self-healing.")

        # 3. History Table & CSV Export
        st.divider()
        st.subheader("Saved Records Table")
        
        # Filter table data
        filtered_df = all_df.copy()
        if selected_battery != "All Batteries":
            filtered_df = filtered_df[filtered_df["battery_id"] == selected_battery]
        if "search_query" in locals() and search_query:
            filtered_df = filtered_df[filtered_df["battery_id"].str.contains(search_query, case=False, na=False)]

        st.dataframe(
            filtered_df[[
                "id", "timestamp", "battery_id", "cycle", "soh", 
                "condition", "v_mean", "v_min", "t_mean", "t_max", "remaining_life"
            ]],
            use_container_width=True,
            hide_index=True
        )

        # CSV Export
        csv_buffer = io.StringIO()
        filtered_df.to_csv(csv_buffer, index=False)
        csv_bytes = csv_buffer.getvalue().encode("utf-8")

        export_c1, export_c2 = st.columns([2, 2])
        with export_c1:
            st.download_button(
                label="📥 Download History as CSV",
                data=csv_bytes,
                file_name=f"voltsense_history_{selected_battery.lower().replace(' ', '_')}.csv",
                mime="text/csv",
                key="csv_download_btn"
            )

        # 4. Delete Record Section
        with export_c2:
            with st.expander("🗑️ Delete a Record"):
                record_ids = filtered_df["id"].tolist()
                if record_ids:
                    del_id = st.selectbox("Select Record ID to delete:", options=record_ids)
                    confirm_del = st.checkbox(f"Confirm deletion of Record #{del_id}", key="confirm_del_box")
                    if st.button("Delete Selected Record", key="delete_record_btn"):
                        if confirm_del:
                            if delete_prediction(del_id):
                                st.success(f"Record #{del_id} deleted successfully. Please refresh the page.")
                                st.rerun()
                            else:
                                st.error("Failed to delete record.")
                        else:
                            st.warning("Please check the confirmation box before deleting.")
                else:
                    st.write("No records to delete.")

    # 5. Scientific Limitations Footer
    st.divider()
    st.info(
        "🔬 **Experimental Limitations & Scientific Caveats:**\n\n"
        "- This model was trained on 4 NASA laboratory lithium-ion batteries under controlled cycling.\n"
        "- Predictions have not been validated on real-world electric vehicle packs with dynamic drive cycles.\n"
        "- Individual SOH predictions have an experimental uncertainty of about ±3 SOH points.\n"
        "- Historical trends and remaining-life estimates are mathematical approximations and do not substitute for physical safety inspections or BMS diagnostics."
    )
