import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

df = pd.read_csv("data/processed/battery_capacity.csv")
df["soh_rel"] = df.groupby("battery_id")["capacity_ah"].transform(lambda s: s / s.iloc[0] * 100)

for held_out in sorted(df["battery_id"].unique()):
    train = df[df["battery_id"] != held_out]
    test = df[df["battery_id"] == held_out]
    model = LinearRegression().fit(train[["cycle"]], train["soh_rel"])
    mae = mean_absolute_error(test["soh_rel"], model.predict(test[["cycle"]]))
    print("Held out:", held_out, "| MAE (SOH points):", round(mae, 3))
