import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

train = pd.read_csv("fashion_train.csv")
test = pd.read_csv("fashion_test.csv")

# Separate labels and pixel values
y_train = train["y"]
X_train = train.drop("y", axis=1)

y_test = test["y"]
X_test = test.drop("y", axis=1)

# Use smaller sample
X_train = X_train.iloc[:3000]
y_train = y_train.iloc[:3000]
X_test = X_test.iloc[:1000]
y_test = y_test.iloc[:1000]

# Scale pixel values
X_train = X_train / 255.0
X_test = X_test / 255.0

# Compare three SVM kernels
for kernel in ["linear", "poly", "rbf"]:
    model = SVC(kernel=kernel)
    model.fit(X_train, y_train)

    prediction = model.predict(X_test)
    accuracy = accuracy_score(y_test, prediction)

    print(kernel, "kernel accuracy:", accuracy)
