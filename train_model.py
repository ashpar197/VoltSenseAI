import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

df = pd.read_csv("data/processed/battery_capacity.csv")

train = df[df["battery_id"] != "B0018"]
test = df[df["battery_id"] == "B0018"]

X_train, y_train = train[["cycle"]], train["capacity_ah"]
X_test, y_test = test[["cycle"]], test["capacity_ah"]

dummy = DummyRegressor(strategy="mean").fit(X_train, y_train)
linear = LinearRegression().fit(X_train, y_train)

print("Mean-guess MAE (Ah):      ", round(mean_absolute_error(y_test, dummy.predict(X_test)), 4))
print("Linear Regression MAE (Ah):", round(mean_absolute_error(y_test, linear.predict(X_test)), 4))
print("B0018 capacity range (Ah): ", round(y_test.min(), 3), "to", round(y_test.max(), 3))
