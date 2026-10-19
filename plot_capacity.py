import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/processed/battery_capacity.csv")

for name, g in df.groupby("battery_id"):
    plt.plot(g["cycle"], g["capacity_ah"], label=name)

plt.xlabel("Cycle")
plt.ylabel("Capacity (Ah)")
plt.title("Capacity fade by battery")
plt.legend()
plt.savefig("capacity_plot.png", dpi=100)
print("Saved capacity_plot.png")
