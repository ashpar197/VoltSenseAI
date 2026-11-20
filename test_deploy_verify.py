import sys
from pathlib import Path
import pandas as pd
import numpy as np
import joblib

print("--- Test 1: Testing imports from requirements.txt ---")
import streamlit
import pandas
import numpy
import sklearn
import joblib
import matplotlib
import altair
print(f"All imports successful! Streamlit={streamlit.__version__}, scikit-learn={sklearn.__version__}, pandas={pandas.__version__}")

print("\n--- Test 2: Testing model loading & prediction ---")
BASE_DIR = Path(".").resolve()
model_path = BASE_DIR / "models" / "voltsense.joblib"
assert model_path.exists(), f"Model file missing at {model_path}"
bundle = joblib.load(model_path)
assert "model" in bundle and "features" in bundle, "Model bundle structure invalid"
print("Features required:", bundle["features"])

# Test prediction with typical input
sample_input = pd.DataFrame([[50, 3.50, 2.70, 32.0, 38.0]], columns=bundle["features"])
pred_soh = float(bundle["model"].predict(sample_input)[0])
print(f"Prediction successful! Predicted SOH: {pred_soh:.2f}%")
assert 0 <= pred_soh <= 120, "Unreasonable SOH value"

print("\n--- Test 3: Testing recommendation engine ---")
from recommend import recommend
rec = recommend(pred_soh)
assert "condition" in rec and "suitable" in rec and "unsuitable" in rec
print(f"Recommendation successful! Condition: {rec['condition']}, Suitable items: {len(rec['suitable'])}")

print("\n--- Test 4: Testing PDF report generation ---")
from pdf_report import generate_pdf_report
pdf_bytes = generate_pdf_report(
    {"cycle": 50, "v_mean": 3.50, "v_min": 2.70, "t_mean": 32.0, "t_max": 38.0},
    pred_soh,
    rec
)
assert len(pdf_bytes) > 1000 and pdf_bytes.startswith(b"%PDF"), "Invalid PDF generated"
print(f"PDF report generated successfully! Size: {len(pdf_bytes)} bytes")

print("\n--- Test 5: Testing dataset access for Fade Curves ---")
data_path = BASE_DIR / "data" / "processed" / "battery_features.csv"
assert data_path.exists(), f"Dataset missing at {data_path}"
df_curves = pd.read_csv(data_path)
assert "battery_id" in df_curves and "capacity_ah" in df_curves and "cycle" in df_curves
print(f"Dataset verified! Total rows: {len(df_curves)}, Batteries: {df_curves['battery_id'].unique().tolist()}")

print("\n--- Test 6: Testing RUL estimation logic ---")
synthetic_soh = [100.0 - i * 0.3 for i in range(45)]
n = len(synthetic_soh)
x = np.arange(1, n + 1)
slope, intercept = np.polyfit(x[-20:], np.array(synthetic_soh[-20:]), 1)
assert slope < 0, "Expected negative slope"
at80 = (80 - intercept) / slope
left = at80 - n
print(f"RUL estimation successful! Cycles until 80%: {left:.1f}")

print("\n--- Test 7: Testing database functionality ---")
from database import init_db, save_prediction, get_all_predictions, get_battery_summary, delete_prediction
init_db()
test_id = save_prediction("Cloud-Deploy-Check", pred_soh, rec["condition"], 50, 3.50, 2.70, 32.0, 38.0, "Test-OK")
summ = get_battery_summary("Cloud-Deploy-Check")
assert summ["latest_soh"] == round(pred_soh, 2)
df_all = get_all_predictions(battery_id="Cloud-Deploy-Check")
assert len(df_all) >= 1
del_res = delete_prediction(test_id)
assert del_res
print("Database operations (insert, query, summary, delete) all verified!")

print("\n==========================================")
print(">>> ALL LOCAL VERIFICATION CHECKS PASSED <<<")
print("==========================================")
