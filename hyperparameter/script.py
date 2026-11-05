import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# 1. Load dataset
train = pd.read_csv("fashion_train.csv")
test = pd.read_csv("fashion_test.csv")

# 2. Separate features and labels
X_train = train.drop("y", axis=1).iloc[:5000] / 255.0
y_train = train["y"].iloc[:5000]

X_test = test.drop("y", axis=1).iloc[:1000] / 255.0
y_test = test["y"].iloc[:1000]

# 3. Hyperparameter tuning
for rate in [0.001, 0.01]:
    for batch in [32, 128]:
        for epoch in [10, 20]:

            # Create neural network
            model = MLPClassifier(
                hidden_layer_sizes=(100,),
                learning_rate_init=rate,
                batch_size=batch,
                max_iter=epoch,
                random_state=42
            )

            # Train the model
            model.fit(X_train, y_train)

            # Predict and calculate accuracy
            prediction = model.predict(X_test)
            accuracy = accuracy_score(y_test, prediction)

            # Display results
            print("Learning Rate:", rate)
            print("Batch Size:", batch)
            print("Epochs:", epoch)
            print("Accuracy:", accuracy)
            print("---------------------")