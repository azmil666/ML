import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC

# 1. Load dataset
data = pd.read_csv("Iris.csv")

# 2. Select features
X = data[["SepalLengthCm", "PetalLengthCm"]].values

# 3. Binary classification
y = (data["Species"] == "Iris-setosa").astype(int)

# 4. Train Linear SVM
model = SVC(kernel="linear", C=1)
model.fit(X, y)

# 5. Create grid
xx, yy = np.meshgrid(
    np.linspace(X[:,0].min()-1, X[:,0].max()+1, 100),
    np.linspace(X[:,1].min()-1, X[:,1].max()+1, 100)
)

grid = np.c_[xx.ravel(), yy.ravel()]

# 6. Predict
Z = model.predict(grid).reshape(xx.shape)

# 7. Plot decision regions
plt.contourf(xx, yy, Z, alpha=0.3, cmap="coolwarm")

# 8. Plot margins and decision boundary
scores = model.decision_function(grid).reshape(xx.shape)
plt.contour(xx, yy, scores, levels=[-1, 0, 1],
            colors=["black", "red", "black"])

# 9. Plot data and support vectors
plt.scatter(X[:,0], X[:,1], c=y, cmap="coolwarm", edgecolors="black")

plt.scatter(model.support_vectors_[:,0],
            model.support_vectors_[:,1],
            s=120, facecolors="none", edgecolors="green",
            label="Support Vectors")

plt.xlabel("Sepal Length")
plt.ylabel("Petal Length")
plt.title("Linear SVM - Iris Classification")
plt.legend()
plt.show()
