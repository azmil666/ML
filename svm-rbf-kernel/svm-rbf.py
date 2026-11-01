import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Load the CSV files
train = pd.read_csv("fashion_train.csv")
test = pd.read_csv("fashion_test.csv")

# Separate labels and pixel values
y_train = train["label"]
X_train = train.drop("label", axis=1)

y_test = test["label"]
X_test = test.drop("label", axis=1)

# Use a smaller sample so the SVMs run faster
X_train = X_train.iloc[:3000]
y_train = y_train.iloc[:3000]
X_test = X_test.iloc[:1000]
y_test = y_test.iloc[:1000]

# Scale pixel values from 0-255 to 0-1
X_train = X_train / 255.0
X_test = X_test / 255.0

# Compare three SVM kernels
for kernel in ["linear", "poly", "rbf"]:
    model = SVC(kernel=kernel)
    model.fit(X_train, y_train)

    prediction = model.predict(X_test)
    accuracy = accuracy_score(y_test, prediction)

    print(kernel, "kernel accuracy:", accuracy)