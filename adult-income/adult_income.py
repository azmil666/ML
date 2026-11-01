import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 1. Load and clean the CSV
data = pd.read_csv("adult_income.csv")
data.columns = data.columns.str.strip()

# Treat '?' as missing, remove extra spaces, and drop incomplete rows
data = data.replace("?", pd.NA)
for column in data.select_dtypes(include="object").columns:
    data[column] = data[column].str.strip()
data = data.dropna()

# Separate features and income label
X = data.drop("income", axis=1)
y = data["income"].str.replace(".", "", regex=False)

# Identify numeric and categorical columns
numeric_columns = X.select_dtypes(include="number").columns
category_columns = X.select_dtypes(exclude="number").columns

# Preprocess numbers and categories
preprocess = ColumnTransformer([
    ("numbers", StandardScaler(), numeric_columns),
    ("categories", OneHotEncoder(handle_unknown="ignore"), category_columns)
])

# 2. Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Create and train both models
logistic = make_pipeline(
    preprocess,
    LogisticRegression(max_iter=1000)
)

tree = make_pipeline(
    preprocess,
    DecisionTreeClassifier(criterion="entropy", max_depth=5, random_state=42)
)

logistic.fit(X_train, y_train)
tree.fit(X_train, y_train)

# 4. Compare performance
for name, model in [("Logistic Regression", logistic),
                    ("Decision Tree", tree)]:
    prediction = model.predict(X_test)
    print(name)
    print("Accuracy:", accuracy_score(y_test, prediction))
    print("Precision:", precision_score(y_test, prediction, pos_label=">50K"))
    print("Recall:", recall_score(y_test, prediction, pos_label=">50K"))
    print("F1-score:", f1_score(y_test, prediction, pos_label=">50K"))
    print()