import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

desired_width = 320

pd.set_option('display.width', desired_width)

# np.set_printoption(linewidth=desired_width)

pd.set_option('display.max_columns', 10)

data = pd.read_csv("Accidents.csv", sep=',', error_bad_lines=False, index_col=False, dtype={})
columns_names = data.columns.tolist()

# print(columns_names)

# give me filter columns of data
columns_for_pca = data[
    ['Day_of_Week', 'Police_Force', 'Accident_Severity', 'Number_of_Vehicles', 'Number_of_Casualties', 'Speed_limit',
     'Light_Conditions', 'Weather_Conditions']];

# print(columns_for_pca.shape)
# print(columns_for_pca.dtypes)
# print(columns_for_pca.head())
columns_for_pca.apply(pd.to_numeric, errors='ignore')
# print(columns_for_pca.dtypes)

print(columns_for_pca.corr())

# correlation = columns_for_pca.corr(method='pearson')
# plt.figure(figsize=(10,10))
# sns.heatmap(correlation, vmax=1, square=True,annot=True,cmap='cubehelix')

# plt.title('Correlation between different features')
# plt.show()

# print (correlation)

cols = columns_for_pca.columns.tolist()
columns_for_pca = columns_for_pca.reindex(columns=cols)
# print(columns_for_pca)

X = columns_for_pca
from sklearn.preprocessing import StandardScaler

X_std = StandardScaler().fit_transform(X)

mean_vec = np.mean(X_std, axis=0)
cov_mat = (X_std - mean_vec).T.dot((X_std - mean_vec)) / (X_std.shape[0] - 1)
print('Covariance matrix \n%s' % cov_mat)
# print('NumPy covariance matrix: \n%s' %np.cov(X_std.T))
# print(columns_for_pca['Accident_Severity'].corr(columns_for_pca['Number_of_Vehicles']))

# plt.figure(figsize=(8,8))
# sns.heatmap(cov_mat, vmax=1, square=True,annot=True,cmap='cubehelix')

# plt.title('Correlation between different features')
# plt.show()

eig_vals, eig_vecs = np.linalg.eig(cov_mat)

# print('Eigenvectors \n%s' %eig_vecs)
# print('\nEigenvalues \n%s' %eig_vals)

pca = PCA().fit(X)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance');
plt.show();
