import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score

# 1. Load dataset
data = pd.read_csv("online-retail.csv")

# 2. Preprocess data
data = data.dropna(subset=["CustomerID"])
data = data[(data["Quantity"] > 0) & (data["UnitPrice"] > 0)]

# 3. Calculate total amount
data["Amount"] = data["Quantity"] * data["UnitPrice"]

# 4. Group data customer-wise
customers = data.groupby("CustomerID").agg({
    "InvoiceNo": "nunique",
    "Amount": "sum",
    "Quantity": "sum",
    "StockCode": "nunique"
}).reset_index()

# Rename columns
customers.columns = [
    "CustomerID", "invoice_count", "total_spend",
    "total_quantity", "different_products"
]

# 5. Create customer segments
customers["Segment"] = pd.qcut(
    customers["invoice_count"],
    q=3,
    labels=["Low", "Medium", "High"],
    duplicates="drop"
)

# 6. Select input and output
X = customers[[
    "total_spend", "total_quantity", "different_products"
]]
y = customers["Segment"]

# 7. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 8. Create and train Decision Tree
model = DecisionTreeClassifier(
    criterion="entropy",
    max_depth=3
)
model.fit(X_train, y_train)

# 9. Predict and calculate accuracy
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# 10. Display feature importance
print("\nFeature Importance:")
for name, value in zip(X.columns, model.feature_importances_):
    print(name, ":", round(value, 3))

# 11. Visualize Decision Tree
plt.figure(figsize=(12, 6))
plot_tree(
    model,
    feature_names=X.columns,
    class_names=model.classes_,
    filled=True
)
plt.show()
