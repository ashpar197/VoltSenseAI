import numpy as np
import pandas as pd

df = pd.read_csv("data/processed/battery_capacity.csv")

for name, g in df.groupby("battery_id"):
    slope = np.polyfit(g["cycle"], g["capacity_ah"], 1)[0]
    print(name,
          "| cycles:", len(g),
          "| first Ah:", round(g["capacity_ah"].iloc[0], 3),
          "| last Ah:", round(g["capacity_ah"].iloc[-1], 3),
          "| slope (Ah per cycle):", round(slope, 5))
