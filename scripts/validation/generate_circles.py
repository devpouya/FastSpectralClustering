import numpy as np
from sklearn.datasets import make_blobs, make_circles

k = int(input("enter desired number of clusters: "))
n = int(input("enter desired number of total datasets points: "))


points, y = make_blobs(n_samples=n,centers=k,cluster_std=0.5,n_features=2,random_state=7)

#points, y = make_circles(n_samples=n)

print(points.shape)


file = open("datasets/test_points/%s_c.txt"%k, "w+")
file.write("2\n")
for i in range(n):
    for j in range(2):
        if j != 1:
            file.write(str(points[i][j]) + " ")
        else:
            file.write(str(points[i][j]) + "\n")
