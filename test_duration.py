import numpy as np
import pandas as pd
from scipy.io import loadmat
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

df = pd.read_csv("data/processed/battery_features.csv")
df["soh_rel"] = df.groupby("battery_id")["capacity_ah"].transform(lambda s: s / s.iloc[0] * 100)

dur = []
for name in ["B0005", "B0006", "B0007", "B0018"]:
    for c in loadmat(f"data/raw/{name}.mat", simplify_cells=True)[name]["cycle"]:
        if c["type"] == "discharge":
            dur.append(float(np.atleast_1d(c["data"]["Time"])[-1]))
df["duration"] = dur

print("Correlation duration vs capacity:", round(df["duration"].corr(df["capacity_ah"]), 3))

base = ["cycle", "v_mean", "v_min", "t_mean", "t_max"]
for label, cols in [("without duration", base), ("with duration", base + ["duration"])]:
    scores = []
    for h in sorted(df["battery_id"].unique()):
        tr, te = df[df["battery_id"] != h], df[df["battery_id"] == h]
        m = RandomForestRegressor(n_estimators=100, random_state=42).fit(tr[cols], tr["soh_rel"])
        scores.append(mean_absolute_error(te["soh_rel"], m.predict(te[cols])))
    print(label, "| per battery:", [round(s, 2) for s in scores], "| average:", round(sum(scores) / 4, 2))
