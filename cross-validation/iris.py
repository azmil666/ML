import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

# Load Iris data
iris = load_iris()
X = iris.data
y = iris.target

# Use the same model for both resampling methods
model = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=1000)
)

# 1. Bootstrap evaluation
rng = np.random.default_rng(42)
bootstrap_accuracy = []
bootstrap_f1 = []

for i in range(100):
    # Sample rows with replacement
    train_indices = rng.choice(len(X), size=len(X), replace=True)

    # Use rows not selected in the sample as the test set
    selected = np.zeros(len(X), dtype=bool)
    selected[train_indices] = True
    test_indices = np.where(~selected)[0]

    # Skip unusual samples with no test rows
    if len(test_indices) == 0:
        continue

    model.fit(X[train_indices], y[train_indices])
    prediction = model.predict(X[test_indices])

    bootstrap_accuracy.append(accuracy_score(y[test_indices], prediction))
    bootstrap_f1.append(
        f1_score(y[test_indices], prediction, average="weighted")
    )

# 2. Five-fold stratified cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

results = cross_validate(
    model,
    X,
    y,
    cv=cv,
    scoring=["accuracy", "f1_weighted"]
)

# 3. Compare mean scores
print("Bootstrap mean accuracy:", np.mean(bootstrap_accuracy))
print("Bootstrap mean F1-score:", np.mean(bootstrap_f1))

print("\n5-fold CV mean accuracy:", results["test_accuracy"].mean())
print("5-fold CV mean F1-score:", results["test_f1_weighted"].mean())