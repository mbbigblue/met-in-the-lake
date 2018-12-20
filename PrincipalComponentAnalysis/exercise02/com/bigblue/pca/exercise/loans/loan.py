# Imports
import numpy as np
import pandas as pd
import seaborn as sns
sns.set(font_scale=1.2, style="whitegrid")  # set styling preferences

## Download loan.gz from https://www.dropbox.com/s/pvpvm4y9gygsas2/loan.csv.gz?dl=0 and extract to the same directory as loan.py

## 1. Load the loan data and take a look on them
loan = pd.read_csv('loan.csv').sample(frac=.25)  # read the dataset and sample 25% of it
loan.replace([np.inf, -np.inf], np.nan)  # convert infs to nans
loan = loan.dropna(axis=1, how='any')  # remove nans
loan = loan._get_numeric_data()  # keep only numeric features
print(loan.head())
columns_names = loan.columns.tolist()
print(columns_names)

## 2. Standaridize and calculate covariance matrix
# X = loan.values #convert the data into a numpy array
# X = np.scale(X)
# covar_matrix = PCA(n_components = 20) #take 20 features
# covar_matrix.fit(X)
# covariance = covar_matrix.get_covariance()
# print(covariance)

# ## 3. Plot Covariance Heatmap
# plt.figure(figsize=(20, 20))
# sns.heatmap(covariance, vmax=1, square=False, annot=True, cmap=sns.color_palette("Blues", 20))
# plt.title('Correlation between different features')
# plt.show()
#
# ## 4. Calculate eigenvalues & eigenvectors
# variance = covar_matrix.explained_variance_ratio_
# cummulativeVariance=np.cumsum(np.round(covar_matrix.explained_variance_ratio_, decimals=3))
# print(cummulativeVariance)
#
# eig_vals, eig_vecs = np.linalg.eig(covariance)
# print('\nEigenvalues \n%s' % eig_vals)
# print('Eigenvectors \n%s' % eig_vecs)
# standard_eigenValues = eig_vals / sum(eig_vals)
# print('Standardize Eigenvalues \n%s', standard_eigenValues)
#
#
# ## 5. Plot heatmap of the Covariance Matrix
# plt.ylabel('% Variance Explained')
# plt.xlabel('# of Components')
# plt.title('PCA Analysis')
# plt.style.context('Blues')
# plt.plot(cummulativeVariance)
#
# ## 6. Scree Chart
# sing_vals = np.arange(len(standard_eigenValues)) + 1
# plt.plot(sing_vals, np.cumsum(standard_eigenValues), 'ro-', linewidth=2)
# plt.title('Scree Plot')
# plt.xlabel('Principal Components')
# plt.ylabel('Standardized Eigenvalues')
# plt.show()
#
# ## 7. Taking 10 components
# pca = PCA(n_components=10)
# pca.fit_transform(X)
# print pca.explained_variance_ratio_
# print(sum(pca.explained_variance_ratio_))
# print('Columns recommended: ', columns_names[:10])
