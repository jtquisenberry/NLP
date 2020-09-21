# Load libraries
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

# Load iris data
iris = load_iris()

# Create features and target
X = iris.data
y = iris.target

# Convert to categorical data by converting data to integers
X = X.astype(int)


# Select two features with highest chi-squared statistics
chi2_selector = SelectKBest(chi2, k=2)
X_kbest = chi2_selector.fit_transform(X, y)


# Show results
print('Original number of features:', X.shape[1])
print('Reduced number of features:', X_kbest.shape[1])