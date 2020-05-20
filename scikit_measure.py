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
num_runs = 1
median_idx = 0

# First Benchmark: growing dim
test = "growing dim"
output_filename = "vec_8.txt"
dataset_path = os.getcwd() + "/benchmarks/datasets/6c_5000n_growing_dim/"
output_path = os.getcwd() + "/benchmarks/kmeans/measurements/"
k = 6
n = 5000
params = [2, 4, 8, 16, 32, 64, 128, 256, 512]

# # Second Benchmark: growing n
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
performances_median = []


# dataset_path = "datasets/test_points/"+input("enter the dataset file name (from datasets/test_points/): ")
# number_clusters = input("enter number of clusters  ")
# output_path = "output/validation/"+input("enter the output file name (result in /output/validation/): ")
#

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
        # print("filename: {}".format(filename))
        if filename == str(par) + ".txt":
            # compute NUM_RUNS times and get the median
            runtimes = []
            for i in range(0, num_runs):
                data_np = load_data(dataset_path + filename)
                t1_start = perf_counter()
                clustering_result, time_ar = \
                                    SpectralClustering(n_clusters=k, eigen_solver='arpack', n_components=k, random_state=30,
                                                       n_init=1, gamma=1.0, affinity='rbf',  eigen_tol=1e-12,
                                                       assign_labels='kmeans', kernel_params=None, n_jobs=1, kmeans_algo='full').fit_predict(data_np)
                t1_stop = perf_counter()
                runtime_sec =  t1_stop-t1_start-time_ar
                runtimes.append(runtime_sec)
            # sort the arrays
            runtimes.sort()
            # adding to final list
            runtimes_median.append(runtimes[median_idx])

            print("runtime: "+ str(runtimes[median_idx]) +" (sec)")


with open(str(output_path) + output_filename + "_runtime", 'w') as f:
    writer = csv.writer(f, delimiter='\t')
    writer.writerows(zip(params, runtimes_median))



