import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC

# Load the CSV
data = pd.read_csv("Iris.csv")

# Use two features so we can plot the boundary
X = data[["SepalLengthCm", "PetalLengthCm"]].values

# Setosa = 1, other species = 0
y = (data["Species"] == "Iris-setosa").astype(int)

# Train linear SVM
model = SVC(kernel="linear", C=1)
model.fit(X, y)

# Create a grid for plotting
x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
xx, yy = np.meshgrid(
    np.arange(x_min, x_max, 0.02),
    np.arange(y_min, y_max, 0.02)
)

# Get model scores across the grid
grid = np.c_[xx.ravel(), yy.ravel()]
scores = model.decision_function(grid).reshape(xx.shape)

# Plot decision regions, boundary, and margins
plt.contourf(xx, yy, scores > 0, alpha=0.2, cmap="coolwarm")
plt.contour(xx, yy, scores, levels=[-1, 0, 1],
            colors=["black", "red", "black"])

# Plot flowers and support vectors
plt.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", edgecolors="black")
plt.scatter(
    model.support_vectors_[:, 0],
    model.support_vectors_[:, 1],
    s=120, facecolors="none", edgecolors="green",
    label="Support vectors"
)

plt.xlabel("Sepal length (cm)")
plt.ylabel("Petal length (cm)")
plt.title("Linear SVM: Setosa vs. Non-Setosa")
plt.legend()
plt.show()