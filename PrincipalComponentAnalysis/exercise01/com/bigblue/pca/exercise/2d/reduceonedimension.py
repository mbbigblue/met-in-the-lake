import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns;

sns.set()

##1. Generate random data
rng = np.random.RandomState(1)
X = np.dot(rng.rand(2, 2), rng.randn(2, 200)).T
colors = list(np.random.choice(range(5), size=200))
# Plot generated data
plt.scatter(X[:, 0], X[:, 1], c=colors, alpha=0.8)
plt.axis('equal')

## 2. PCA estimation
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
pca.fit(X)

print(pca.components_)  # representing the directions of maximum variance in the data
print(pca.explained_variance_)  # The amount of variance explained by each of the selected components


## 3. Draw vector method
def draw_vector(v0, v1, ax=None):
    ax = ax or plt.gca()
    arrowprops = dict(arrowstyle='->',
                      linewidth=2,
                      shrinkA=0, shrinkB=0)
    ax.annotate('', v1, v0, arrowprops=arrowprops)


## draw a chart
plt.scatter(X[:, 0], X[:, 1], alpha=0.2)
for length, vector in zip(pca.explained_variance_, pca.components_):
    v = vector * 3 * np.sqrt(length)
    draw_vector(pca.mean_, pca.mean_ + v)
plt.axis('equal');

## 4. It's time for dimensionality reduction!
pca = PCA(n_components=1)
pca.fit(X)
X_pca = pca.transform(X)
print("original shape:   ", X.shape)
print("transformed shape:", X_pca.shape)

## 5. Getting back the data

X_new = pca.inverse_transform(X_pca)
plt.scatter(X[:, 0], X[:, 1], alpha=0.2)
plt.scatter(X_new[:, 0], X_new[:, 1], alpha=0.8)
plt.axis('equal')

plt.show()

# pca = PCA(n_components=1)
# pca.fit(X)
# X_pca = pca.transform(X)
# print("original shape:   ", X)
# print("transformed shape:", X_pca)
#
# X_new = pca.inverse_transform(X_pca)
# plt.scatter(X[:, 0], X[:, 1], alpha=0.3)
# plt.scatter(X_new[:, 0], X_new[:, 1], alpha=0.6, color='red')
# plt.axis('equal');
plt.show()
