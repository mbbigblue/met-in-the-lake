import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
desired_width = 320
pd.set_option('display.width', desired_width)
pd.set_option('display.max_columns', 10)

## 1. Load the accidents data and take a look on them
data = pd.read_csv("Accidents.csv", sep=',', error_bad_lines=False, index_col=False, dtype='unicode')
columns_names = data.columns.tolist()

print(columns_names)
# print(data)

## 2. Pick up the columns for analysis
columns_for_pca = data[
    ['Day_of_Week', 'Police_Force', 'Accident_Severity', 'Number_of_Vehicles', 'Number_of_Casualties', 'Speed_limit',
     'Light_Conditions', 'Weather_Conditions']];

print(columns_for_pca.shape)
print(columns_for_pca.head())
columns_for_pca.apply(pd.to_numeric, errors='ignore')


cols = columns_for_pca.columns.tolist()
columns_for_pca = columns_for_pca.reindex(columns=cols)
print(columns_for_pca)

## 3. Standardize and Calculate Covariance Matrix
X = columns_for_pca
from sklearn.preprocessing import StandardScaler

X_std = StandardScaler().fit_transform(X)

mean_vec = np.mean(X_std, axis=0)
cov_mat = (X_std - mean_vec).T.dot((X_std - mean_vec)) / (X_std.shape[0] - 1)
print('Covariance matrix \n%s' % cov_mat)
print('NumPy covariance matrix: \n%s' % np.cov(X_std.T))

## 4. Plot heatmap of the Covariance Matrix
plt.figure(figsize=(8, 8))
sns.heatmap(cov_mat, vmax=1, square=True, annot=True, cmap=sns.color_palette("Blues", 80))

plt.title('Correlation between different features')
plt.show()

## 5. Calculate eng_vals, eig_vec
eig_vals, eig_vecs = np.linalg.eig(cov_mat)
print('\nEigenvalues \n%s' % eig_vals)
print('Eigenvectors \n%s' % eig_vecs)
standard_eigenValues = eig_vals / sum(eig_vals)
print('Standardize Eigenvalues \n%s', standard_eigenValues)

## 6. Run PCA
pca = PCA().fit(X)
variance = np.round(pca.explained_variance_ratio_, decimals=4) * 100
plt.plot(np.cumsum(variance))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')
plt.show()

## 7. Scree Chart

sing_vals = np.arange(len(standard_eigenValues)) + 1
plt.plot(sing_vals, np.cumsum(standard_eigenValues), 'ro-', linewidth=2)
plt.title('Scree Plot')
plt.xlabel('Principal Components')
plt.ylabel('Standardized Eigenvalues')
plt.show()

# pca = PCA(n_components=4)
# pca.fit(X)
# X1=pca.fit_transform(X)
#
# print X1

## Taking 6 components
pca = PCA(n_components=6)
pca.fit_transform(X)
print pca.explained_variance_ratio_
print(sum(pca.explained_variance_ratio_))
