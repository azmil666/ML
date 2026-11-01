import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. Load and preprocess
data = pd.read_csv("housing.csv")
data = data[["total_rooms", "households", "median_house_value"]].dropna()
data = data[data["households"] != 0]

# One feature: average rooms per household
x = (data["total_rooms"] / data["households"]).values
y = data["median_house_value"].values

# Scale x so gradient descent works reliably
x = (x - x.mean()) / x.std()

# Add a column of 1s for the intercept
X = np.c_[np.ones(len(x)), x]
n = len(y)

# 2. Gradient descent
theta = np.zeros(2)
for i in range(1000):
    error = X @ theta - y
    theta = theta - 0.01 * (2 / n) * (X.T @ error)

prediction_gd = X @ theta

# 3. Normal equation
theta2 = np.linalg.pinv(X.T @ X) @ X.T @ y
prediction_ne = X @ theta2

# 4. Print MSE and R-squared for both methods
for name, prediction in [("Gradient descent", prediction_gd),
                         ("Normal equation", prediction_ne)]:
    mse = np.mean((y - prediction) ** 2)
    r2 = 1 - np.sum((y - prediction) ** 2) / np.sum((y - y.mean()) ** 2)
    print(name, "MSE:", mse, "R-squared:", r2)

# 5. Plot data and fitted line
order = np.argsort(x)
plt.scatter(x, y, s=5)
plt.plot(x[order], prediction_ne[order], color="red")
plt.xlabel("Average rooms per household (scaled)")
plt.ylabel("Median house value")
plt.show()