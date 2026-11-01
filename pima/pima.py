import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Column names for the raw dataset
columns = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"
]

# Load the CSV (no header row in this file)
data = pd.read_csv("diabetes.csv", header=None, names=columns)

# Separate input features and target
X = data.drop("Outcome", axis=1)
y = data["Outcome"]

# Use the same train/test split for both models
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Model 1: without feature scaling
model1 = LogisticRegression(max_iter=1000)
model1.fit(X_train, y_train)
prediction1 = model1.predict(X_test)

# Model 2: with feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model2 = LogisticRegression(max_iter=1000)
model2.fit(X_train_scaled, y_train)
prediction2 = model2.predict(X_test_scaled)

# Print evaluation metrics
for name, prediction in [
    ("Without scaling", prediction1),
    ("With scaling", prediction2)
]:
    print(name)
    print("Accuracy:", accuracy_score(y_test, prediction))
    print("Precision:", precision_score(y_test, prediction, zero_division=0))
    print("Recall:", recall_score(y_test, prediction, zero_division=0))
    print("F1-score:", f1_score(y_test, prediction, zero_division=0))
    print()