import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load data
data = pd.read_csv("breast-cancer.csv")
data = data.drop(["id", "Unnamed: 32"], axis=1)

# X and y
X = data.drop("diagnosis", axis=1)
y = data["diagnosis"].map({"B": 0, "M": 1})

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# MLE
mle = LogisticRegression(penalty=None, max_iter=5000)

# MAP L2
l2 = LogisticRegression(penalty="l2")

# MAP L1
l1 = LogisticRegression(penalty="l1", solver="liblinear")

# Train and compare
for name, model in [("MLE", mle), ("L2", l2), ("L1", l1)]:
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    print(name, accuracy_score(y_test, pred))
    print("Coefficients:", model.coef_[0][:5])


"""
import pandas as pd
import sklearn
from packaging.version import Version
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load data
data = pd.read_csv("breast-cancer.csv")
data = data.drop(["id", "Unnamed: 32"], axis=1, errors="ignore")

# X and y
X = data.drop("diagnosis", axis=1)
y = data["diagnosis"].map({"B": 0, "M": 1})

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Models: adapt to sklearn version
if Version(sklearn.__version__) >= Version("1.8"):
    mle = LogisticRegression(C=float("inf"), max_iter=5000)
    l2 = LogisticRegression(l1_ratio=0, max_iter=5000)
    l1 = LogisticRegression(l1_ratio=1, solver="liblinear", max_iter=5000)
else:
    mle = LogisticRegression(penalty="none", max_iter=5000)
    l2 = LogisticRegression(penalty="l2", max_iter=5000)
    l1 = LogisticRegression(penalty="l1", solver="liblinear", max_iter=5000)

# Train and compare
for name, model in [("MLE", mle), ("L2", l2), ("L1", l1)]:
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    print(name, accuracy_score(y_test, pred))
    print("Coefficients:", model.coef_[0][:5])
"""