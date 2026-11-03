import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load CSV files
train = pd.read_csv("fashion_train.csv")
test = pd.read_csv("fashion_test.csv")

# Separate features and target
y_train = train["y"]
X_train = train.drop("y", axis=1) / 255.0

y_test = test["y"]
X_test = test.drop("y", axis=1) / 255.0

# Use fewer images for faster execution
X_train = X_train.iloc[:10000]
y_train = y_train.iloc[:10000]

X_test = X_test.iloc[:2000]
y_test = y_test.iloc[:2000]

# Test different K values
for k in [3, 5, 7]:

    model = KNeighborsClassifier(n_neighbors=k)

    model.fit(X_train, y_train)

    prediction = model.predict(X_test)

    accuracy = accuracy_score(y_test, prediction)

    print("K =", k)
    print("Accuracy:", accuracy)
