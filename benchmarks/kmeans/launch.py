import subprocess
import os
import csv

# global settings
num_runs = 9
median_idx = 4

# First Benchmark: growing dim
test = "growing k"
output_filename = "10-base_kmeans_growing_k"
dataset_path = os.getcwd() + "/benchmarks/datasets/2d_2500n_growing_k/"
output_path = os.getcwd() + "/benchmarks/measurements/"
#k = 6
n = 2500
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
iter = 1
ks = []
for file in sorted(os.listdir(directory)): # when generating data have just numbers for simplicity
        print("RUNNING ITER:%d"%iter)
        filename = os.fsdecode(file)
        k = filename.split(".")[0]
        print(k)
        # compute NUM_RUNS times and get the median
        runtimes = []
        performances = []
        for i in range(0, num_runs):
            clustering = subprocess.check_output(["./clustering", str(dataset_path) + filename, str(k), "out.txt"],
                                                 universal_newlines="\n").split("\n")
            #print(clustering[0])
            runtime = clustering[2]
            runtimes.append(runtime)
            countops = subprocess.check_output(["./countops", str(dataset_path) + filename, str(k), "out.txt"],
                                                 universal_newlines="\n").split("\n")
            #print(countops[1])
            flops = countops[3]
            performances.append(float(flops)/float(runtime))
        # sort the arrays
        runtimes.sort()
        performances.sort()
        # adding to final list
        runtimes_median.append(runtimes[median_idx])
        performances_median.append(performances[median_idx])
        ks.append(k)
        print("runtime: "+ str(runtimes[median_idx]) +" (cycles), performance: "+ str(performances[median_idx]) +" (flops/cycle)")
        iter += 1

with open(str(output_path) + output_filename + "_runtime.txt", 'w') as f:
    writer = csv.writer(f, delimiter='\t')
    writer.writerows(zip(ks, runtimes_median))

with open(str(output_path) + output_filename + "_perf.txt", 'w') as f:
    writer = csv.writer(f, delimiter='\t')
    writer.writerows(zip(ks, performances_median))
