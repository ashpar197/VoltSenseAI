import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

df = pd.read_csv("data/processed/battery_features.csv")
df["soh_rel"] = df.groupby("battery_id")["capacity_ah"].transform(lambda s: s / s.iloc[0] * 100)

sets = {
    "cycle only (Linear)": (["cycle"], LinearRegression()),
    "cycle + 4 features (Linear)": (["cycle", "v_mean", "v_min", "t_mean", "t_max"], LinearRegression()),
    "cycle + 4 features (Forest)": (["cycle", "v_mean", "v_min", "t_mean", "t_max"], RandomForestRegressor(n_estimators=100, random_state=42)),
}

for label, (cols, model) in sets.items():
    scores = []
    for held_out in sorted(df["battery_id"].unique()):
        train = df[df["battery_id"] != held_out]
        test = df[df["battery_id"] == held_out]
        model.fit(train[cols], train["soh_rel"])
        scores.append(mean_absolute_error(test["soh_rel"], model.predict(test[cols])))
    print(label, "| per battery:", [round(s, 2) for s in scores], "| average:", round(sum(scores) / 4, 2))
