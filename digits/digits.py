from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Load dataset
digits = load_digits()
X = digits.data

# Scale the data
X = StandardScaler().fit_transform(X)

# Try different K values
for k in [2, 5, 10, 15]:

    # Apply K-means
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = model.fit_predict(X)

    # Calculate metrics
    print("K =", k)
    print("Inertia:", model.inertia_)
    print("Silhouette Score:", silhouette_score(X, labels))
    print()