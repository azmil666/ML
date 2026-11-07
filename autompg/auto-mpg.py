import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

data = pd.read_csv("auto-mpg.data", sep=r"\s+", header=None, na_values="?")
data = data.dropna()

# 3. Select input and output
X = data[[2]]       # displacement
y = data[0]         # MPG

# 4. Try different polynomial degrees
for d in [1, 2, 3]:
    poly = PolynomialFeatures(degree=d, include_bias=False)
    X_poly = poly.fit_transform(X)

    model = LinearRegression()
    model.fit(X_poly, y)

    pred = model.predict(X_poly)

    print("Degree:", d)
    print("MSE:", mean_squared_error(y, pred))
    print("R2 Score:", r2_score(y, pred))

    # Plot fitted curve
    order = np.argsort(X.values[:, 0])
    plt.plot(X.values[order, 0], pred[order], label=f"Degree {d}")

# 5. Plot original data
plt.scatter(X, y, color="blue", label="Actual Data")
plt.xlabel("Displacement")
plt.ylabel("MPG")
plt.title("Polynomial Regression")
plt.legend()
plt.show()
