import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import subprocess
import random
import os

dataset_path = os.getcwd()+"/datasets/test_points/"+input("enter the dataset file name (from datasets/test_points/): ")
number_clusters = input("enter number of clusters  ")
output_path = os.getcwd()+"/output/validation/"+input("enter the output file name (result in /output/validation/): ")

subprocess.run(["make"])
subprocess.run(["./clustering", dataset_path, number_clusters, output_path])


f = open(output_path, "r")

# read the cluster number k  (actually redundant)
f.readline()

# read the sizes of each clusters
cluster_sizes = f.readline()
cluster_sizes = [int(i) for i in cluster_sizes.strip().split(" ")]
# print("cluster sizes are : ")
# print(cluster_sizes)

# read the original dataset
number_of_points = sum(cluster_sizes)
print("in total {} points".format(number_of_points))
f_dataset = open(dataset_path, "r")
dim = int(f_dataset.readline())
data = np.zeros((number_of_points, dim))
line_idx = 0
for lines in f_dataset:
    line_data = lines.strip().split()
    # print(line_data)
    for i in range(dim):
        data[line_idx][i] = line_data[i]
    line_idx = line_idx + 1
# print("datasets set shape :{}".format(data.shape))

# this is a list, which stores points by clusters
clustered_data = []
for k in range(int(number_clusters)):
    point_index = [int(i) for i in f.readline().strip().split()]
    # assuming dim = 2 here
    assert dim == 2
    x = np.zeros(cluster_sizes[k])
    y = np.zeros(cluster_sizes[k])
    for i in range(cluster_sizes[k]):
        x[i] = data[point_index[i]][0]
        y[i] = data[point_index[i]][1]
    clustered_data.append([x.copy(), y.copy()])

# print(len(clustered_data[0][0]))
# print(len(clustered_data[1][0]))
# print(clustered_data[0][0])
# print(clustered_data[0][1])

# plot the clusters
number_of_colors = int(number_clusters)
color = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
             for i in range(number_of_colors)]
# print(color)
for i in range(int(number_clusters)):
    plt.scatter(clustered_data[i][0], clustered_data[i][1], color=color[i])

plt.show()
