from pathlib import Path
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

df = pd.read_csv("data/processed/battery_features.csv")
df["soh_rel"] = df.groupby("battery_id")["capacity_ah"].transform(lambda s: s / s.iloc[0] * 100)

cols = ["cycle", "v_mean", "v_min", "t_mean", "t_max"]
model = RandomForestRegressor(n_estimators=100, random_state=42).fit(df[cols], df["soh_rel"])

Path("models").mkdir(exist_ok=True)
joblib.dump({"model": model, "features": cols}, "models/voltsense.joblib")
print("Saved models/voltsense.joblib")
