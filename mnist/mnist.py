import time
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# Load the labeled MNIST CSV
data = pd.read_csv("mnist_train.csv")

# Separate labels and pixel values
X = data.drop("label", axis=1)
y = data["label"]

# Use a smaller sample to make the comparison faster
X = X.iloc[:10000]
y = y.iloc[:10000]

# Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale pixel values
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train and compare the three activation functions
for activation in ["logistic", "relu", "tanh"]:
    model = MLPClassifier(
        hidden_layer_sizes=(100,),
        activation=activation,
        max_iter=20,
        random_state=42
    )

    start = time.time()
    model.fit(X_train, y_train)
    seconds = time.time() - start

    prediction = model.predict(X_test)

    print("Activation:", activation)
    print("Accuracy:", accuracy_score(y_test, prediction))
    print("Training time (seconds):", round(seconds, 2))
    print("Iterations used:", model.n_iter_)
    print()