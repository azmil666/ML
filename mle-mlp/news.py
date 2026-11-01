import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

# Load the CSV from the same folder as this Python file
data = pd.read_csv("train_preprocessed.csv")
documents = data["raw_text"].fillna("").astype(str)

# Convert documents into word counts
vectorizer = CountVectorizer(max_features=1000)
word_counts = vectorizer.fit_transform(documents)

# Count each word across all documents
counts = np.asarray(word_counts.sum(axis=0)).ravel()
words = vectorizer.get_feature_names_out()
total = counts.sum()
vocab_size = len(words)

# MLE: word count divided by total word count
mle = counts / total

# MAP with two Dirichlet prior values
top_words = np.argsort(counts)[-5:][::-1]

for alpha in [0.1, 1.0]:
    map_estimate = (counts + alpha) / (total + alpha * vocab_size)

    print("\nPrior alpha:", alpha)
    print("Word       MLE       MAP")

    for i in top_words:
        print(f"{words[i]:10s} {mle[i]:.6f}  {map_estimate[i]:.6f}")