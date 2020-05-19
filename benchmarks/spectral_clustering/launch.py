import subprocess
import os
import csv

# global settings
num_runs = 9
median_idx = 4

# First Benchmark: growing dim
test = "growing n"
output_filename = "vec_8.txt"
dataset_path = os.getcwd() + "/benchmarks/datasets/growing_n/"
output_path = os.getcwd() + "/benchmarks/spectral_clustering/measurements/"
k = 6
dim = 2
params = range(100, 5000, 100)

# global conf
# subprocess.run(["make", "countops"])
subprocess.run(["make"])
directory = os.fsencode(str(dataset_path))
times_median = []

for par in params:
    print(test + " | parameter = " + str(par))
    for file in sorted(os.listdir(directory)): # when generating data have just numbers for simplicity
        filename = os.fsdecode(file)
        if filename == str(par) + ".txt":
            # compute NUM_RUNS times and get the median
            times = []
            for i in range(0, num_runs):
                clustering = subprocess.check_output(["./clustering", str(dataset_path) + filename, str(k), "out.txt"],
                                                 universal_newlines="\n").split("\n")
                #print(clustering[0])
                time = clustering[0]
                times.append(time)

            # sort the arrays
            times.sort()

            # adding to final list
            times_median.append(times[median_idx])
            print(" time: " + str(times[median_idx]) +" (sec)")


with open(str(output_path) + output_filename + "_time", 'w') as f:
    writer = csv.writer(f, delimiter='\t')
    writer.writerows(zip(params, times_median))

