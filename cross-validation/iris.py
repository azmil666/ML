import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

# 1. Load Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# 2. Create model with preprocessing
model = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=1000)
)

# 3. Bootstrap evaluation
rng = np.random.default_rng(42)
boot_acc = []
boot_f1 = []

for i in range(100):

    # Random sampling with replacement
    train = rng.choice(len(X), len(X), replace=True)

    # Find samples not selected
    selected = np.zeros(len(X), dtype=bool)
    selected[train] = True
    test = np.where(~selected)[0]

    # Train and test
    model.fit(X[train], y[train])
    pred = model.predict(X[test])

    # Calculate metrics
    boot_acc.append(accuracy_score(y[test], pred))
    boot_f1.append(f1_score(y[test], pred, average='weighted'))

# 4. K-fold cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

results = cross_validate(
    model, X, y, cv=cv,
    scoring=['accuracy', 'f1_weighted']
)

# 5. Compare results
print("Bootstrap Accuracy:", np.mean(boot_acc))
print("Bootstrap F1-score:", np.mean(boot_f1))

print("Cross Validation Accuracy:", results['test_accuracy'].mean())
print("Cross Validation F1-score:", results['test_f1_weighted'].mean())
