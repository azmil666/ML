import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

# 1. Load dataset
data = pd.read_csv("train_preprocessed.csv")
documents = data["raw_text"].fillna("")

# 2. Convert text into word counts
vectorizer = CountVectorizer(max_features=1000)
X = vectorizer.fit_transform(documents)

counts = np.asarray(X.sum(axis=0)).ravel()
words = vectorizer.get_feature_names_out()

total = counts.sum()
V = len(words)

# 3. MLE estimation
mle = counts / total

# 4. MAP estimation with different Dirichlet priors
top = np.argsort(counts)[-5:][::-1]

for alpha in [1.1, 2, 10]:

    map_estimate = (counts + alpha - 1) / (total + V * (alpha - 1))

    print("\nAlpha =", alpha)
    print("Word       MLE       MAP")

    for i in top:
        print(f"{words[i]:10s} {mle[i]:.6f}  {map_estimate[i]:.6f}")

