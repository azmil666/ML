from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Load the dataset
digits = load_digits()
X = digits.data

# Scale the pixel features
X = StandardScaler().fit_transform(X)

# Try different numbers of clusters
for k in [2, 5, 10, 15]:
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = model.fit_predict(X)

    inertia = model.inertia_
    silhouette = silhouette_score(X, labels)

    print("K =", k)
    print("Inertia:", inertia)
    print("Silhouette score:", silhouette)
    print()