
Exercise 01
1. Generate random data
Problem looks like linear regression, but we are not going to predict y in next x. We want to discover relationship between x, y.

2. Use the PCA to find components and explained_variance(components" to define the direction of the vector, explained variance" to define the squared-length of the vector)


#These vectors represent the principal axes of the data, and the length of the vector is an indication of how "important"
#that axis is in describing the distribution of the data—more precisely, it is a measure of the variance of the data
# when projected onto that axis. The projection of each data point onto the principal axes are the "principal components" of the data.


#The light points are the original data, while the dark points are the projected version. This makes clear what a PCA dimensionality reduction means: the information along the least important principal axis or axes is removed, leaving only the component(s) of the data with the highest variance. The fraction of variance that is cut out (proportional to the spread of points about the line formed in this figure) is roughly a measure of how much "information" is discarded in this reduction of dimensionality.

#This reduced-dimension dataset is in some senses "good enough" to encode the most important relationships between the points: despite reducing the dimension of the data by 50%, the overall relationship between the data points are mostly preserved.
#
#