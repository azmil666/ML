import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.metrics import accuracy_score, f1_score

# Load the dataset
data = pd.read_csv("train_preprocessed.csv")

# Get message text and topic labels
X = data["raw_text"].fillna("").astype(str)
y = data["target"]

# Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Convert text into word counts
count_vectorizer = CountVectorizer(stop_words="english", max_features=5000)
X_train_counts = count_vectorizer.fit_transform(X_train)
X_test_counts = count_vectorizer.transform(X_test)

# Convert counts into word presence (0 or 1)
binary_vectorizer = CountVectorizer(
    stop_words="english", max_features=5000, binary=True
)
X_train_binary = binary_vectorizer.fit_transform(X_train)
X_test_binary = binary_vectorizer.transform(X_test)

# Train both classifiers
multinomial = MultinomialNB()
multinomial.fit(X_train_counts, y_train)

bernoulli = BernoulliNB()
bernoulli.fit(X_train_binary, y_train)

# Predict topics
prediction_multi = multinomial.predict(X_test_counts)
prediction_bern = bernoulli.predict(X_test_binary)

# Compare accuracy and F1-score
print("Multinomial Naive Bayes")
print("Accuracy:", accuracy_score(y_test, prediction_multi))
print("F1-score:", f1_score(y_test, prediction_multi, average="weighted"))

print("\nBernoulli Naive Bayes")
print("Accuracy:", accuracy_score(y_test, prediction_bern))
print("F1-score:", f1_score(y_test, prediction_bern, average="weighted"))