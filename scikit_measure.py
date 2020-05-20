import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import subprocess
import random
import os
from sklearn.cluster import SpectralClustering
from time import perf_counter
import subprocess
import os
import csv

# global settings
num_runs = 3
median_idx = 1

# # First Benchmark: growing dim
# test = "growing dim"
# output_filename = ""
# dataset_path = os.getcwd() + "/benchmarks/datasets/6c_5000n_growing_dim/"
# output_path = os.getcwd() + "/benchmarks/scikit/measurements/"
# k = 6
# n = 5000
#
# params = [2, 4]

# second Benchmark: growing k
test = "growing k"
output_filename = ""
dataset_path = os.getcwd() + "/benchmarks/datasets/2d_2500n_growing_k/"
output_path = os.getcwd() + "/benchmarks/scikit/measurements/growing_k/"

n = 2500
# params = [2]
params = [i for i in range(2,100)]


# # Third Benchmark: growing n
# test = "growing n"
# output_filename = "vec_8.txt"
# dataset_path = os.getcwd() + "benchmarks/datasets/8c_256d_growing_n/"
# output_path = os.getcwd() + "/benchmarks/graph/measurements/"
# k = 6 #??
# dim = 2 #??
# n = range(10, 10000, 100)

# global

directory = os.fsencode(str(dataset_path))
runtimes_median = []
elkan_runtimes_median = []
total_runtimes_median = []

def load_data(dataset_path):
    f_input = open(dataset_path, "r")
    data = f_input.read().split()
    dim = int(data[0])
    data = data[1:]
    data_np = np.array(data).reshape(-1, dim)
    return data_np

for par in params:
    print(test + " | parameter = " + str(par))
    for file in sorted(os.listdir(directory)): # when generating data have just numbers for simplicity
        filename = os.fsdecode(file)

        if filename == str(par) + ".txt":
            # compute NUM_RUNS times and get the median
            runtimes = []
            # runtimes2 = []
            # runtimes3 = []
            k = par
            print("number of clusters : {}".format(k))
            for i in range(0, num_runs):
                data_np = load_data(dataset_path + filename)
                # t1_start = perf_counter()
                clustering_result, time_kmeans = \
                                    SpectralClustering(n_clusters=k, eigen_solver='arpack', n_components=k, random_state=30,
                                                       n_init=1, gamma=1.0, affinity='rbf',  eigen_tol=1e-12,
                                                       assign_labels='kmeans', kernel_params=None, n_jobs=1, kmeans_algo='full').fit_predict(data_np)
                # print(clustering_result)
                # print(clustering_result.shape)
                # t1_stop = perf_counter()
                runtimes.append(time_kmeans)
                # runtimes3.append(t1_stop-t1_start)

            # sort the arrays
            runtimes.sort()
            # runtimes2.sort()
            # runtimes3.sort()
            # adding to final list
            runtimes_median.append(runtimes[median_idx])
            # elkan_runtimes_median.append(runtimes2[median_idx])
            # total_runtimes_median.append(runtimes3[median_idx])

            print("runtime lloyd kmeans part: "+ str(runtimes[median_idx]) +" (sec)")
            # print("runtime elkan no_eig: " + str(runtimes2[median_idx]) + " (sec)")
            # print("runtime lloyd with_eig: " + str(runtimes3[median_idx]) + " (sec)")
            # print("runtime eig: " + str(runtimes3[median_idx] - runtimes[median_idx]) + " (sec)")


with open(str(output_path)+ "runtime_lloyd_kmeans_part", 'w') as f:
    writer = csv.writer(f, delimiter='\t')
    writer.writerows(zip(params, runtimes_median))
#
# with open(str(output_path) + "_elkan_runtime_no_eig", 'w') as f:
#     writer = csv.writer(f, delimiter='\t')
#     writer.writerows(zip(params, elkan_runtimes_median))
#
# with open(str(output_path) + "_total_lloyd_runtime_with_eig", 'w') as f:
#     writer = csv.writer(f, delimiter='\t')
#     writer.writerows(zip(params, total_runtimes_median))
#
#

