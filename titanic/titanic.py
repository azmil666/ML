import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 1. Load the CSV and choose useful columns
data = pd.read_csv("train.csv")

X = data[["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]]
y = data["Survived"]

# 2. Prepare numeric and categorical columns
numeric_columns = ["Pclass", "Age", "SibSp", "Parch", "Fare"]
category_columns = ["Sex", "Embarked"]

preprocess = ColumnTransformer([
    ("numbers", SimpleImputer(strategy="median"), numeric_columns),
    ("categories", make_pipeline(
        SimpleImputer(strategy="most_frequent"),
        OneHotEncoder(handle_unknown="ignore")
    ), category_columns)
])

# Split once so both models use the same test passengers
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Preprocess features
X_train = preprocess.fit_transform(X_train)
X_test = preprocess.transform(X_test)

# 3. Bagging: combine many decision trees
bagging = BaggingClassifier(
    estimator=DecisionTreeClassifier(),
    n_estimators=100,
    random_state=42
)
bagging.fit(X_train, y_train)

# 4. Boosting: AdaBoost combines trees sequentially
boosting = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1),
    n_estimators=100,
    learning_rate=0.5,
    random_state=42
)
boosting.fit(X_train, y_train)

# 5. Compare performance
for name, model in [("Bagging", bagging), ("AdaBoost", boosting)]:
    prediction = model.predict(X_test)

    print(name)
    print("Accuracy:", accuracy_score(y_test, prediction))
    print("Precision:", precision_score(y_test, prediction, zero_division=0))
    print("Recall:", recall_score(y_test, prediction, zero_division=0))
    print("F1-score:", f1_score(y_test, prediction, zero_division=0))
    print()