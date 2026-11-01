import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score

# 1. Load and preprocess
data = pd.read_csv("diabetes.csv")
data = data.dropna()

X = data.drop("target", axis=1)
y = data["target"]

# Use the same training and test rows for every model
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 2. Standard linear regression
linear = make_pipeline(StandardScaler(), LinearRegression())
linear.fit(X_train, y_train)
linear_prediction = linear.predict(X_test)

# 3. Tune Ridge using 5-fold cross-validation
ridge = GridSearchCV(
    make_pipeline(StandardScaler(), Ridge()),
    {"ridge__alpha": [0.01, 0.1, 1, 10, 100]},
    cv=5
)
ridge.fit(X_train, y_train)
ridge_prediction = ridge.predict(X_test)

# 4. Tune Lasso using 5-fold cross-validation
lasso = GridSearchCV(
    make_pipeline(StandardScaler(), Lasso(max_iter=10000)),
    {"lasso__alpha": [0.01, 0.1, 1, 10, 100]},
    cv=5
)
lasso.fit(X_train, y_train)
lasso_prediction = lasso.predict(X_test)

# 5. Compare MSE and R-squared
for name, prediction in [
    ("Linear Regression", linear_prediction),
    ("Ridge", ridge_prediction),
    ("Lasso", lasso_prediction)
]:
    print(name)
    print("MSE:", mean_squared_error(y_test, prediction))
    print("R-squared:", r2_score(y_test, prediction))
    print()

print("Best Ridge alpha:", ridge.best_params_)
print("Best Lasso alpha:", lasso.best_params_)