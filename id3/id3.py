import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 1. Load and clean the CSV
data = pd.read_csv("online-retail.csv")
data = data.dropna(subset=["CustomerID"])
data = data[(data["Quantity"] > 0) & (data["UnitPrice"] > 0)]

# Calculate the amount spent per row
data["Amount"] = data["Quantity"] * data["UnitPrice"]

# 2. Make one row per customer
customers = data.groupby("CustomerID").agg(
    invoice_count=("InvoiceNo", "nunique"),
    total_spend=("Amount", "sum"),
    average_order=("Amount", "mean"),
    total_quantity=("Quantity", "sum"),
    different_products=("StockCode", "nunique")
).reset_index()

# Create customer segments from purchase frequency
customers["Segment"] = pd.qcut(
    customers["invoice_count"],
    q=3,
    labels=["Low", "Medium", "High"],
    duplicates="drop"
)

# Use purchasing features to predict the segment
X = customers[
    ["total_spend", "average_order", "total_quantity", "different_products"]
]
y = customers["Segment"]

# 3. Split data and build an entropy-based decision tree
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

tree = DecisionTreeClassifier(
    criterion="entropy",
    max_depth=3,
    random_state=42
)
tree.fit(X_train, y_train)

# 4. Evaluate
prediction = tree.predict(X_test)
print("Accuracy:", accuracy_score(y_test, prediction))

# 5. Print feature importance
print("\nFeature importance:")
for name, importance in zip(X.columns, tree.feature_importances_):
    print(name, ":", round(importance, 3))

# 6. Plot the decision tree
plt.figure(figsize=(14, 7))
plot_tree(
    tree,
    feature_names=X.columns,
    class_names=[str(label) for label in tree.classes_],
    filled=True,
    rounded=True
)
plt.show()