import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score

# Load dataset
data = pd.read_csv("Mall_Customers.csv")
# Select features
X = data[["Age", "Annual Income (k$)", "Spending Score (1-100)"]]
# Scale the data
X = StandardScaler().fit_transform(X)

# K-means clustering
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
kmeans_labels = kmeans.fit_predict(X)

# Hierarchical clustering
hc = AgglomerativeClustering(n_clusters=5, linkage="ward")
hc_labels = hc.fit_predict(X)

# Evaluation
print("K-means")
print("Inertia:", kmeans.inertia_)
print("Silhouette Score:", silhouette_score(X, kmeans_labels))

print("\nHierarchical Clustering")
print("Silhouette Score:", silhouette_score(X, hc_labels))

# Visualization
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(data["Annual Income (k$)"],
            data["Spending Score (1-100)"], c=kmeans_labels)
plt.title("K-means Clustering")
plt.xlabel("Annual Income")
plt.ylabel("Spending Score")

plt.subplot(1, 2, 2)
plt.scatter(data["Annual Income (k$)"],
            data["Spending Score (1-100)"], c=hc_labels)
plt.title("Hierarchical Clustering")
plt.xlabel("Annual Income")
plt.ylabel("Spending Score")
plt.show()
