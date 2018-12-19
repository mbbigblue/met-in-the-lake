
Exercise 01
1. Generate random data
Problem looks like linear regression, but we are not going to predict y in next x. We want to discover relationship between x, y.
2. Use the PCA to find components and explained_variance(components" to define the direction of the vector, explained variance" to define the squared-length of the vector)
These vectors represent the principal axes of the data, and the length of the vector is an indication of how "important"

3. PCA dimensionality reduction means:
 - the information along the least important principal axis or axes is removed,
 - leaving only the component(s) of the data with the highest variance.
 - The fraction of variance that is cut out (proportional to the spread of points about the line formed in this figure)
  is roughly a measure of how much "information" is discarded in this reduction of dimensionality.


Exercise 02
1. Review accidents data
2.



Worth to know:

 - PCA is used to overcome features redundancy in a data set.
 - These features are low dimensional in nature.
 - These features a.k.a components are a resultant of normalized linear combination of original predictor variables.
 - These components aim to capture as much information as possible with high explained variance.
 - The first component has the highest variance followed by second, third and so on.
 - The components must be uncorrelated (remember orthogonal direction ? ). See above.
 - Normalizing data becomes extremely important when the predictors are measured in different units.
 - PCA works best on data set having 3 or higher dimensions. Because, with higher dimensions, it becomes increasingly difficult to make interpretations from the resultant cloud of data.
 - PCA is applied on a data set with numeric variables.
 - PCA is a tool which helps to produce better visualizations of high dimensional data.