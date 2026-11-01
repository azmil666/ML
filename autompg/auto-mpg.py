import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Column names for the raw UCI file
columns = [
    "mpg", "cylinders", "displacement", "horsepower", "weight",
    "acceleration", "year", "origin", "car_name"
]

# Load the raw file (it is space-separated, not a CSV)
data = pd.read_csv(
    "auto-mpg.data",
    sep=r"\s+",
    names=columns,
    na_values="?"
)

# Clean missing values
data = data.dropna()

# Use displacement to predict MPG
X = data[["displacement"]]
y = data["mpg"]

# Compare linear and polynomial models
for degree in [1, 2, 3]:
    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(X)

    model = LinearRegression()
    model.fit(X_poly, y)
    prediction = model.predict(X_poly)

    print("Degree:", degree)
    print("MSE:", mean_squared_error(y, prediction))
    print("R-squared:", r2_score(y, prediction))
    print()

    # Plot this fitted curve
    order = np.argsort(X["displacement"])
    plt.plot(
        X["displacement"].values[order],
        prediction[order],
        label=f"Degree {degree}"
    )

# Plot data points and fitted lines
plt.scatter(X["displacement"], y, s=15, color="blue", label="Data")
plt.xlabel("Engine displacement")
plt.ylabel("Miles per gallon (MPG)")
plt.legend()
plt.show()