import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load the CSV
data = pd.read_csv("BostonHousing.csv")

# Use average rooms to predict median home value
X = data[["RM"]]
y = data["MEDV"]

# Split into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.3, random_state=42
)

degrees = range(1, 11)
train_errors = []
val_errors = []

# Fit polynomial regression models of different degrees
for degree in degrees:
    model = make_pipeline(
        StandardScaler(),
        PolynomialFeatures(degree=degree),
        LinearRegression()
    )

    model.fit(X_train, y_train)

    train_prediction = model.predict(X_train)
    val_prediction = model.predict(X_val)

    train_errors.append(mean_squared_error(y_train, train_prediction))
    val_errors.append(mean_squared_error(y_val, val_prediction))

# Print errors
for degree, train_error, val_error in zip(degrees, train_errors, val_errors):
    print("Degree:", degree,
          "| Training MSE:", round(train_error, 2),
          "| Validation MSE:", round(val_error, 2))

# Plot training and validation errors
plt.plot(degrees, train_errors, marker="o", label="Training error")
plt.plot(degrees, val_errors, marker="o", label="Validation error")
plt.xlabel("Polynomial degree")
plt.ylabel("Mean Squared Error")
plt.title("Bias-Variance Tradeoff")
plt.xticks(list(degrees))
plt.legend()
plt.show()