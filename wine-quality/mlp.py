import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score

# 1. Load and preprocess the dataset
data = pd.read_csv("winequality-red.csv", sep=";")

X = data.drop("quality", axis=1)
y = data["quality"]

# Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale feature values for the neural network
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 2. Compare networks with different hidden layers and neuron counts
architectures = [
    (10,),          # one hidden layer, 10 neurons
    (50,),          # one hidden layer, 50 neurons
    (50, 25)         # two hidden layers, 50 and 25 neurons
]

for layers in architectures:
    model = MLPRegressor(
        hidden_layer_sizes=layers,
        max_iter=1000,
        random_state=42
    )

    model.fit(X_train, y_train)
    prediction = model.predict(X_test)

    print("Hidden layers:", layers)
    print("MSE:", mean_squared_error(y_test, prediction))
    print("R-squared:", r2_score(y_test, prediction))
    print()