import numpy as np
import pandas as pd

df = pd.read_csv("data/processed/battery_features.csv")
df["soh"] = df.groupby("battery_id")["capacity_ah"].transform(lambda s: s / s.iloc[0] * 100)

for name, g in df.groupby("battery_id"):
    c, s = g["cycle"].to_numpy(), g["soh"].to_numpy()
    true = int(c[s < 80][0])
    for n in [20, 40, 60]:
        if n >= true:
            print(name, "| known cycles:", n, "| skipped (already past 80%)")
            continue
        w = slice(max(0, n - 20), n)  # trend from the last 20 known cycles
        slope, intercept = np.polyfit(c[w], s[w], 1)
        if slope >= 0:
            print(name, "| known cycles:", n, "| no estimate (no downward trend)")
            continue
        pred = (80 - intercept) / slope
        print(name, "| known cycles:", n, "| predicted 80% at cycle:", round(pred),
              "| true:", true, "| error (cycles):", round(pred - true))
