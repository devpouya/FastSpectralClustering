import numpy as np
import sys
from sklearn.datasets import make_blobs

if (len(sys.argv) != 4):
    k = int(input("enter desired number of clusters: "))
    n = int(input("enter desired number of total datasets points: "))
    out = input("enter output filename: ")
else:
    k = int(sys.argv[1])
    n = int(sys.argv[2])
    out = sys.argv[3]

dim = 128
points, y = make_blobs(n_samples=n,centers=k,n_features=dim,random_state=7)

"""
means = np.random.choice(n, (k,1,2), replace=False)
print(means)
cov = [[1, 0], [0, 1]]

g = np.random.gamma(range(k+1),k+1)
g /= np.sum(g)
g *= n
g = np.rint(g)[1:]
print(g)

points = np.empty((n,2))
for i in range(k):
    st = np.random.multivariate_normal(means[i].reshape(2,), cov, int(g[i]))
    points = np.append(points,st,axis=0)
np.random.shuffle(points)
"""
file = open("/datasets/test_points/"+out, "w+")
file.write("%d\n" % dim)
for i in range(n):
    for j in range(dim):
        if j != dim-1:
            file.write(str(points[i][j]) + " ")
        else:
            file.write(str(points[i][j]) + "\n")