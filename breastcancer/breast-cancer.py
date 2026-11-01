import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 1. Load and preprocess
data = pd.read_csv("breast-cancer.csv")

# Remove ID column if present
if "id" in data.columns:
    data = data.drop("id", axis=1)

# Remove rows with missing values
data = data.dropna()

# Features and diagnosis
X = data.drop("Diagnosis", axis=1)
y = data["Diagnosis"].map({"B": 0, "M": 1})

# Split data for fair comparison
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 2. MLE: logistic regression without regularization
mle = make_pipeline(
    StandardScaler(),
    LogisticRegression(penalty=None, max_iter=5000)
)
mle.fit(X_train, y_train)

# 3. MAP with L2 regularization
l2 = make_pipeline(
    StandardScaler(),
    LogisticRegression(penalty="l2", C=1.0, max_iter=5000)
)
l2.fit(X_train, y_train)

# 4. MAP with L1 regularization
l1 = make_pipeline(
    StandardScaler(),
    LogisticRegression(penalty="l1", solver="liblinear", C=1.0, max_iter=5000)
)
l1.fit(X_train, y_train)

# 5. Compare performance
for name, model in [("MLE", mle), ("MAP L2", l2), ("MAP L1", l1)]:
    prediction = model.predict(X_test)
    print(name, "accuracy:", accuracy_score(y_test, prediction))

# 6. Compare parameter estimates (coefficients)
print("\nCoefficient comparison:")
print("MLE:   ", mle[-1].coef_[0][:5])
print("MAP L2:", l2[-1].coef_[0][:5])
print("MAP L1:", l1[-1].coef_[0][:5])