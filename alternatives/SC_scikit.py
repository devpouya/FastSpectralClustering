from sklearn.cluster import KMeans
import numpy as np

A = np.array([
    [0, 1, 1, 0, 0, 0, 0, 0, 1, 1],
    [1, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 1, 1, 0, 1, 1, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 1, 0]])

# our adjacency matrix
print("Adjacency Matrix:")
print(A)

# Adjacency Matrix:
# [[0. 1. 1. 0. 0. 1. 0. 0. 1. 1.]
#  [1. 0. 1. 0. 0. 0. 0. 0. 0. 0.]
#  [1. 1. 0. 0. 0. 0. 0. 0. 0. 0.]
#  [0. 0. 0. 0. 1. 1. 0. 0. 0. 0.]
#  [0. 0. 0. 1. 0. 1. 0. 0. 0. 0.]
#  [1. 0. 0. 1. 1. 0. 1. 1. 0. 0.]
#  [0. 0. 0. 0. 0. 1. 0. 1. 0. 0.]
#  [0. 0. 0. 0. 0. 1. 1. 0. 0. 0.]
#  [1. 0. 0. 0. 0. 0. 0. 0. 0. 1.]
#  [1. 0. 0. 0. 0. 0. 0. 0. 1. 0.]]

# diagonal matrix
D = np.diag(A.sum(axis=1))

# graph laplacian
L = D-A

# eigenvalues and eigenvectors
vals, vecs = np.linalg.eig(L)

# sort these based on the eigenvalues
vecs = vecs[:,np.argsort(vals)]
vals = vals[np.argsort(vals)]

# kmeans on first three vectors with nonzero eigenvalues
kmeans = KMeans(n_clusters=4)
kmeans.fit(vecs[:,1:4])
colors = kmeans.labels_

print("Clusters:", colors)

# Clusters: [2 1 1 0 0 0 3 3 2 2]