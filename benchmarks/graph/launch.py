import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import subprocess
import random
import os
import sys
import csv

# global settings
num_runs = 1
median_idx = 0

# First Benchmark: growing dim
test = "growing dim"
output_filename = "vec_8.txt"
dataset_path = os.getcwd() + "/benchmarks/datasets/6c_5000n_growing_dim/"
output_path = os.getcwd() + "/benchmarks/graph/measurements/"
k = 6
n = 5000
params = [2, 4, 8, 16, 32, 64, 128, 256, 512]

# # Second Benchmark: growing n
# test = "growing n"
# output_filename = "vec_8.txt"
# dataset_path = os.getcwd() + "benchmarks/datasets/growing_n/"
# output_path = os.getcwd() + "/benchmarks/graph/measurements/"
# k = 6 #??
# dim = 2 #??
# n = range(10, 10000, 100)

# global
subprocess.run(["make", "countops"])
subprocess.run(["make"])
directory = os.fsencode(str(dataset_path))
runtimes_median = []
performances_median = []

for par in params:
    print(test + " | parameter = " + str(par))
    for file in sorted(os.listdir(directory)): # when generating data have just numbers for simplicity
        filename = os.fsdecode(file)
        if filename == str(par) + ".txt":
            # compute NUM_RUNS times and get the median
            runtimes = []
            performances = []
            for i in range(0, num_runs):
                clustering = subprocess.check_output(["./clustering", str(dataset_path) + filename, str(k), "out.txt"],
                                                 universal_newlines="\n").split("\n")
                #print(clustering[0])
                runtime = clustering[0]
                runtimes.append(runtime)
                countops = subprocess.check_output(["./countops", str(dataset_path) + filename, str(k), "out.txt"],
                                                 universal_newlines="\n").split("\n")
                #print(countops[1])
                flops = countops[1]
                performances.append(float(runtime)/float(flops))
            # sort the arrays
            runtimes.sort()
            performances.sort()
            # adding to final list
            runtimes_median.append(runtimes[median_idx])
            performances_median.append(performances[median_idx])
            print("runtime: "+ str(runtimes[median_idx]) +" (cycles), performance: "+ str(performances[median_idx]) +" (flops/cycle)")


with open(str(output_path) + output_filename + "_runtime", 'w') as f:
    writer = csv.writer(f, delimiter='\t')
    writer.writerows(zip(params, runtimes_median))

with open(str(output_path) + output_filename + "_perf", 'w') as f:
    writer = csv.writer(f, delimiter='\t')
    writer.writerows(zip(params, performances_median))

