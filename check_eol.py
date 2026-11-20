import numpy as np
import pandas as pd

df = pd.read_csv("data/processed/battery_features.csv")
df["soh"] = df.groupby("battery_id")["capacity_ah"].transform(lambda s: s / s.iloc[0] * 100)

for name, g in df.groupby("battery_id"):
    c, s = g["cycle"].to_numpy(), g["soh"].to_numpy()
    below = c[s < 80]
    first80 = int(below[0]) if len(below) else None
    # compare fade speed in the first half vs the second half
    h = len(c) // 2
    slope1 = np.polyfit(c[:h], s[:h], 1)[0]
    slope2 = np.polyfit(c[h:], s[h:], 1)[0]
    print(name, "| cycles:", len(c), "| first cycle below 80%:", first80,
          "| slope 1st half:", round(slope1, 3), "| slope 2nd half:", round(slope2, 3))
