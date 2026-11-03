import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score

# 1. Load dataset
data = pd.read_csv("winequality-red.csv", sep=";")

X = data.drop("quality", axis=1)
y = data["quality"]

# 2. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Scale data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 4. Try different architectures
architectures = [(10,), (50,), (50, 25)]

for layers in architectures:

    model = MLPRegressor(
        hidden_layer_sizes=layers,
        max_iter=1000,
        random_state=42
    )

    model.fit(X_train, y_train)

    prediction = model.predict(X_test)

    print("Architecture:", layers)
    print("MSE:", mean_squared_error(y_test, prediction))
    print("R2 Score:", r2_score(y_test, prediction))