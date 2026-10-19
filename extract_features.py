import numpy as np
import pandas as pd
from scipy.io import loadmat

rows = []
for name in ["B0005", "B0006", "B0007", "B0018"]:
    cycles = loadmat(f"data/raw/{name}.mat", simplify_cells=True)[name]["cycle"]
    n = 0
    for c in cycles:
        if c["type"] != "discharge":
            continue
        n += 1
        d = c["data"]
        v = np.atleast_1d(d["Voltage_measured"])
        t = np.atleast_1d(d["Temperature_measured"])
        rows.append({
            "battery_id": name,
            "cycle": n,
            "capacity_ah": float(np.atleast_1d(d["Capacity"])[0]),
            "v_mean": v.mean(),
            "v_min": v.min(),
            "t_mean": t.mean(),
            "t_max": t.max(),
        })

out = pd.DataFrame(rows)
out.to_csv("data/processed/battery_features.csv", index=False)
print("Rows:", len(out))
print("Missing values:", int(out.isna().sum().sum()))
print(out.groupby("battery_id").size())
print(out.head())
