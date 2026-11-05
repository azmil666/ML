import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC

# 1. Load dataset
data = pd.read_csv("Iris.csv")

# 2. Select two features
X = data[["SepalLengthCm", "PetalLengthCm"]].values

# 3. Convert into binary classification
y = (data["Species"] == "Iris-setosa").astype(int)

# 4. Create and train Linear SVM
model = SVC(kernel="linear", C=1)
model.fit(X, y)

# 5. Create grid for decision boundary
x1, x2 = X[:, 0].min()-1, X[:, 0].max()+1
y1, y2 = X[:, 1].min()-1, X[:, 1].max()+1

xx, yy = np.meshgrid(
    np.linspace(x1, x2, 100),
    np.linspace(y1, y2, 100)
)

grid = np.c_[xx.ravel(), yy.ravel()]

# 6. Predict grid points
Z = model.predict(grid).reshape(xx.shape)

# 7. Plot decision regions
plt.contourf(xx, yy, Z, alpha=0.3, cmap="coolwarm")

# 8. Plot decision boundary and margins
scores = model.decision_function(grid).reshape(xx.shape)
plt.contour(xx, yy, scores, levels=[-1, 0, 1],
            colors=["black", "red", "black"])

# 9. Plot data points
plt.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm",
            edgecolors="black")

# 10. Highlight support vectors
plt.scatter(model.support_vectors_[:, 0],
            model.support_vectors_[:, 1],
            s=120, facecolors="none", edgecolors="green",
            label="Support Vectors")

plt.xlabel("Sepal Length")
plt.ylabel("Petal Length")
plt.title("Linear SVM - Iris Classification")
plt.legend()
plt.show()
