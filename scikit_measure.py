import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import subprocess
import random
import os
from sklearn.cluster import SpectralClustering, KMeans
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
# params = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]

# second Benchmark: growing k
test = "growing_k_"
output_filename = ""
dataset_path = os.getcwd() + "/benchmarks/datasets/2d_2500n_growing_k/"
output_path = os.getcwd() + "/benchmarks/scikit/measurements/growing_k/"

n = 2500
# params = [2]
params = [i for i in range(2, 100)]

# # Third Benchmark: growing n
# test = "growing_n_"
# output_filename = ""
# dataset = "72c_300d_growing_n_normalized"
# dataset_path = os.getcwd() + "/benchmarks/datasets/"+dataset+"/"
# output_path = os.getcwd() + "/benchmarks/scikit/measurements/growing_n/"+dataset+"/"
# k = 72 #??
# dim = 300 #??
# params = [n for n in range(100, 6100, 100)]

# global

directory = os.fsencode(str(dataset_path))
runtimes_median = []
elkan_runtimes_median = []
total_runtimes_median = []

def load_data(dataset_path, num_clusters):
    f_input = open(dataset_path, "r")
    data = f_input.read().split()
    # dim = int(data[0])
    # data = data[1:]
    data_np = np.array(data).reshape(-1, num_clusters)
    # data_np = np.array(data).reshape(-1, dim)
    return data_np

for par in params:
    print(test + " | parameter = " + str(par))
    for file in sorted(os.listdir(directory)): # when generating data have just numbers for simplicity
        filename = os.fsdecode(file)

        if filename == str(par) + "_ev.txt":
        # if filename == str(par) + "_ev.txt":
            # compute NUM_RUNS times and get the median
            runtimes = []
            # runtimes2 = []
            # runtimes3 = []
            k = par
            print("number of clusters : {}".format(k))
            data_np = load_data(dataset_path + filename, k)
            for i in range(0, num_runs):
                # print(dataset_path + filename)

                # print("data shape {}".format(data_np.shape))
                # t1_start = perf_counter()
                # t1_graph = SpectralClustering(n_clusters=k, eigen_solver='arpack', n_components=k, random_state=30,
                #                                        n_init=1, gamma=1.0, affinity='rbf',  eigen_tol=1e-12,
                #                                        assign_labels='kmeans', kernel_params=None, n_jobs=1).fit_get_graph_time(data_np)

                runtime = KMeans(n_clusters=k,  init='k-means++', n_init=1, max_iter=1000, tol=0.00000001,
                       precompute_distances=False, verbose=0, random_state=30, copy_x=True,
                       n_jobs=1, algorithm='full').fit_get_runtime(data_np)

                # print(labels)
                # print(clustering_result.shape)
                # t1_stop = perf_counter()
                # print("total time: {}".format(t1_stop-t1_start))
                # runtimes.append(t1_graph)
                runtimes.append(runtime)
                # runtimes3.append(t1_stop-t1_start)

            # sort the arrays
            runtimes.sort()
            # runtimes2.sort()
            # runtimes3.sort()
            # adding to final list
            runtimes_median.append(runtimes[median_idx])
            # elkan_runtimes_median.append(runtimes2[median_idx])
            # total_runtimes_median.append(runtimes3[median_idx])

            print("runtime lloyd part: " + str(runtimes[median_idx]) +" (sec)")
            # print("runtime elkan no_eig: " + str(runtimes2[median_idx]) + " (sec)")
            # print("runtime lloyd with_eig: " + str(runtimes3[median_idx]) + " (sec)")
            # print("runtime eig: " + str(runtimes3[median_idx] - runtimes[median_idx]) + " (sec)")

save_path = str(output_path) + test + "_lloyd_kmeans"
print("saved at :{}".format(save_path))
with open(save_path, 'w') as f:
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

