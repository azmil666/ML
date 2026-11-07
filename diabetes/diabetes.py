
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score

# 1. Load dataset
data = load_diabetes(as_frame=True).frame

X = data.drop("target", axis=1)
y = data["target"]

# 2. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Linear Regression
linear = make_pipeline(StandardScaler(), LinearRegression())
linear.fit(X_train, y_train)
pred1 = linear.predict(X_test)

# 4. Ridge Regression
ridge = GridSearchCV(
    make_pipeline(StandardScaler(), Ridge()),
    {"ridge__alpha": [0.01, 0.1, 1, 10, 100]},
    cv=5
)
ridge.fit(X_train, y_train)
pred2 = ridge.predict(X_test)

# 5. Lasso Regression
lasso = GridSearchCV(
    make_pipeline(StandardScaler(), Lasso(max_iter=10000)),
    {"lasso__alpha": [0.01, 0.1, 1, 10, 100]},
    cv=5
)
lasso.fit(X_train, y_train)
pred3 = lasso.predict(X_test)

# 6. Compare performance
for name, pred in [
    ("Linear Regression", pred1),
    ("Ridge Regression", pred2),
    ("Lasso Regression", pred3)
]:
    print(name)
    print("MSE:", mean_squared_error(y_test, pred))
    print("R2 Score:", r2_score(y_test, pred))
    print()

print("Best Ridge Alpha:", ridge.best_params_)
print("Best Lasso Alpha:", lasso.best_params_)
