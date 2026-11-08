import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score

# 1. Load dataset
data = pd.read_excel("online-retail.xlsx")

# 2. Preprocess
data = data.dropna(subset=["CustomerID"])
data = data[(data["Quantity"] > 0) & (data["UnitPrice"] > 0)]

# 3. Calculate amount
data["Amount"] = data["Quantity"] * data["UnitPrice"]

# 4. Group customer-wise
customers = data.groupby("CustomerID").agg({
    "InvoiceNo": "nunique",
    "Amount": "sum",
    "Quantity": "sum",
    "StockCode": "nunique"
}).reset_index()

customers.columns = ["CustomerID", "invoices", "spend",
                     "quantity", "products"]

# 5. Create segments
customers["Segment"] = pd.qcut(
    customers["invoices"].rank(method="first"),
    q=3, labels=["Low", "Medium", "High"]
)

# 6. Select features and target
X = customers[["spend", "quantity", "products"]]
y = customers["Segment"]

# 7. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 8. Train Decision Tree
model = DecisionTreeClassifier(criterion="entropy", max_depth=3)
model.fit(X_train, y_train)

# 9. Predict and accuracy
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# Feature importance
print(model.feature_importances_)

# Visualize tree
plt.figure(figsize=(12, 6))
plot_tree(model, filled=True)
plt.show()
