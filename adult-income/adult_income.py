import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 1. Load and preprocess data
data = pd.read_csv("adult_income.csv")
data.columns = data.columns.str.strip()
data = data.replace("?", pd.NA)
data = data.dropna()
for col in data.select_dtypes(include="object").columns:
    data[col] = data[col].str.strip()

# Separate features and target
X = data.drop("income", axis=1)
y = data["income"].str.replace(".", "", regex=False)

# Convert categorical columns into numbers
X = pd.get_dummies(X, dtype=int)

# 2. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Scale data for Logistic Regression
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Create and train models
logistic = LogisticRegression(max_iter=1000)
tree = DecisionTreeClassifier(criterion="entropy", max_depth=5)

logistic.fit(X_train_scaled, y_train)
tree.fit(X_train, y_train)

# 5. Compare both models
for name, model, test_data in [
    ("Logistic Regression", logistic, X_test_scaled),
    ("Decision Tree", tree, X_test)
]:
    prediction = model.predict(test_data)

    print("\n", name)
    print("Accuracy:", accuracy_score(y_test, prediction))
    print("Precision:", precision_score(y_test, prediction, pos_label=">50K"))
    print("Recall:", recall_score(y_test, prediction, pos_label=">50K"))
    print("F1-score:", f1_score(y_test, prediction, pos_label=">50K"))
