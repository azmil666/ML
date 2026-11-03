
import time
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# Load CSV files
train = pd.read_csv("fashion_train.csv")
test = pd.read_csv("fashion_test.csv")

# Separate features and target
X_train = train.drop("y", axis=1).iloc[:5000] / 255.0
y_train = train["y"].iloc[:5000]

X_test = test.drop("y", axis=1).iloc[:1000] / 255.0
y_test = test["y"].iloc[:1000]

# Try different hyperparameters
for rate in [0.001, 0.01]:
    for batch in [32, 128]:
        for epochs in [10, 20]:

            model = MLPClassifier(
                hidden_layer_sizes=(100,),
                activation="relu",
                solver="adam",
                learning_rate_init=rate,
                batch_size=batch,
                max_iter=epochs,
                random_state=42
            )

            start = time.time()
            model.fit(X_train, y_train)

            prediction = model.predict(X_test)
            elapsed = time.time() - start

            print("Learning rate:", rate)
            print("Batch size:", batch)
            print("Epochs:", epochs)
            print("Accuracy:", accuracy_score(y_test, prediction))
            print("Training time:", round(elapsed, 2), "seconds")
            print()
