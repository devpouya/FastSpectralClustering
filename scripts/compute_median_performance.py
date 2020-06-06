import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import subprocess
import random
import os
import sys
import csv


dataset_path = os.getcwd()+"/datasets/test_points/"
output_path = os.getcwd() + "/output/measurements/graph_construction_speedup/100c/"
k=6
file_name="6c_5000n_128d.txt"
num_runs=9
dimensions = [2, 4,8, 16,32,64,128,256,512]
subprocess.run(["make"])
directory = os.fsencode(str(dataset_path))

measurements = []

f = file_name
# medians
runtimes = []
for i in range(0, num_runs):
    output = subprocess.check_output(["./clustering", str(dataset_path)+f, str(k), "out.txt"], universal_newlines="\n").split("\n")
    #  print(output[0])
    runtimes.append(output[0])
runtimes.sort()
print(runtimes[4])
#measurements.append(runtimes[4])

#
# with open(str(output_path)+file_name, 'w') as f:
#     writer = csv.writer(f, delimiter='\t')
#     writer.writerows(zip(dimensions, measurements))


