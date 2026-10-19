import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

df = pd.read_csv("data/processed/battery_capacity.csv")
df["soh_rel"] = df.groupby("battery_id")["capacity_ah"].transform(lambda s: s / s.iloc[0] * 100)

train = df[df["battery_id"] != "B0018"]
test = df[df["battery_id"] == "B0018"]

X_train, y_train = train[["cycle"]], train["soh_rel"]
X_test, y_test = test[["cycle"]], test["soh_rel"]

dummy = DummyRegressor(strategy="mean").fit(X_train, y_train)
linear = LinearRegression().fit(X_train, y_train)

print("Mean-guess MAE (SOH points):      ", round(mean_absolute_error(y_test, dummy.predict(X_test)), 3))
print("Linear Regression MAE (SOH points):", round(mean_absolute_error(y_test, linear.predict(X_test)), 3))
print("B0018 SOH range:", round(y_test.min(), 1), "to", round(y_test.max(), 1))
