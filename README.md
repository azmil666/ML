# Machine Learning Examples

This repository contains standalone Python examples of machine learning methods. Each project is kept in its own directory with the script and, where applicable, its dataset. The examples cover regression, classification, clustering, model evaluation, and parameter estimation.

## Repository structure

| Directory | Example |
| --- | --- |
| `adult-income/` | Logistic regression and decision tree classification of income data |
| `autompg/` | Linear and polynomial regression for fuel economy |
| `bostonhousing/` | Polynomial regression and training/validation error comparison |
| `breastcancer/` | Logistic regression with unregularized, L1, and L2 models |
| `cross-validation/` | Bootstrap and stratified cross-validation on the Iris dataset |
| `diabetes/` | Linear, Ridge, and Lasso regression with cross-validation |
| `digits/` | K-means clustering on the scikit-learn digits dataset |
| `housing/` | Linear regression using gradient descent and the normal equation |
| `hyperparameter/` | MLP classification experiments with Fashion-MNIST data |
| `id3/` | Entropy-based decision tree classification of customer segments |
| `iris/` | Linear SVM classification and decision-boundary visualization |
| `knn/` | K-nearest neighbors classification on Fashion-MNIST data |
| `mall-customer/` | K-means and agglomerative clustering of customer data |
| `mle-mlp/` | Word-frequency maximum likelihood and MAP estimates |
| `mnist/` | MLP classification with different activation functions |
| `naive-bayes/` | Multinomial and Bernoulli Naive Bayes for text classification |
| `pima/` | Pima diabetes dataset example |
| `svm-rbf-kernel/` | RBF-kernel SVM classification on Fashion-MNIST data |
| `titanic/` | Bagging and AdaBoost classification |
| `wine-quality/` | MLP regression on red wine quality data |

## Requirements

The scripts use Python and commonly used data science libraries, including pandas, NumPy, Matplotlib, and scikit-learn. Install the libraries required by the example you want to run. For example:

```bash
python -m pip install pandas numpy matplotlib scikit-learn
```

## Running an example

Run a script from its own directory because most scripts open data files using paths relative to the current working directory. For example:

```bash
cd iris
python iris.py
```

Some examples print evaluation results to the terminal. Others also open plots. The scripts are independent; there is no shared training pipeline or command-line interface.

## Data

Data files are stored alongside the scripts that use them. Some examples use bundled CSV or text data, while others load datasets provided by scikit-learn. Check the script for the expected columns and preprocessing steps before substituting another dataset.
