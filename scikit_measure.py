import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import subprocess
import random
import os
from sklearn.cluster import SpectralClustering
from time import perf_counter

dataset_path = "datasets/test_points/"+input("enter the dataset file name (from datasets/test_points/): ")
number_clusters = input("enter number of clusters  ")
output_path = "output/validation/"+input("enter the output file name (result in /output/validation/): ")


f_input = open(dataset_path, "r")

data = f_input.read().split()
dim = int(data[0])
data = data[1:]
data_np = np.array(data).reshape(-1, dim)

number_clusters = int(number_clusters)

t1_start = perf_counter()
for i in range(1):
    clustering_result = \
        SpectralClustering(n_clusters=number_clusters, eigen_solver='arpack', n_components=number_clusters, random_state=30,
                           n_init=1, gamma=1.0, affinity='rbf',  eigen_tol=1e-12,
                           assign_labels='kmeans', kernel_params=None, n_jobs=1, kmeans_algo='full').fit_predict(data_np)
t1_stop = perf_counter()
print("Elapsed time during the whole program in seconds:",
      (t1_stop-t1_start)/1.0)

