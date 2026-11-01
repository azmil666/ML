import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score

# Load the CSV
data = pd.read_csv("Mall_Customers.csv")

# Use age, income, and spending score
X = data[["Age", "Annual Income (k$)", "Spending Score (1-100)"]]

# Scale the features so they are comparable
X_scaled = StandardScaler().fit_transform(X)

# K-means clustering
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
kmeans_labels = kmeans.fit_predict(X_scaled)

# Hierarchical agglomerative clustering
hierarchical = AgglomerativeClustering(n_clusters=5, linkage="ward")
hierarchical_labels = hierarchical.fit_predict(X_scaled)

# Compare clustering metrics
print("K-means")
print("Inertia:", kmeans.inertia_)
print("Silhouette score:", silhouette_score(X_scaled, kmeans_labels))

print("\nHierarchical clustering")
print("Silhouette score:", silhouette_score(X_scaled, hierarchical_labels))

# Plot clusters using income and spending score
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(data["Annual Income (k$)"], data["Spending Score (1-100)"],
            c=kmeans_labels, cmap="viridis")
plt.title("K-means")
plt.xlabel("Annual income (k$)")
plt.ylabel("Spending score")

plt.subplot(1, 2, 2)
plt.scatter(data["Annual Income (k$)"], data["Spending Score (1-100)"],
            c=hierarchical_labels, cmap="viridis")
plt.title("Agglomerative clustering")
plt.xlabel("Annual income (k$)")
plt.ylabel("Spending score")

plt.tight_layout()
plt.show()