import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import subprocess
import random
import os
import sys
import csv

"""
turbo_off = input("Did you remember to turn off Turbo Boost? 1: Yes, 0: No:     ")
if not turbo_off:
    print("Dope")
else:
    print("Okay then, your results will mean nothing!")
    print(turbo_off)

opt_name = input("Which optimization are you testing? (Will be displayed in plot):    ")
"""
num_cluster = input("Number of clusters? (0 for not fixed):    ")
print("Let the fun start!")
subprocess.run(["make"])
directory = os.fsencode("datasets/perf_blobs/growing_dim")


graph_measurements = []
kmeans_measurements = []
n_s = []
i = 1
for file in sorted(os.listdir(directory)):
    filename = os.fsdecode(file)
    if filename.endswith(".txt"):
        #print(filename)
        f = filename
        k= int(filename.split(".")[0])
        output = subprocess.check_output(["./countops", "datasets/perf_blobs/growing_dim/"+f, "5", "out.txt"],universal_newlines="\n").split("\n")
        num_cycle_graph = float(output[2])
        num_ops_graph = float(output[3])
        num_cycle_kmeans = float(output[4])
        num_ops_kmeans = float(output[5])

        #n = int(filename.split(".")[0])
        n_s.append(k)
        print("Iteration %s\n"%i)
        graph_measurements.append(num_ops_graph/num_cycle_graph)
        kmeans_measurements.append(num_ops_kmeans/num_cycle_kmeans)
        i+=1


#sorted_graph = sorted(graph_measurements.keys())
#sorted_kmeans = sorted(kmeans_measurements.keys())
#n_s = sorted_graph.keys()
#perf_graph = sorted_graph.values()
#perf_kmeans = sorted_kmeans.values()

order = np.argsort(n_s)
sorted_graph = np.array(graph_measurements)[order]
sorted_kmeans = np.array(kmeans_measurements)[order]
n_s = np.array(n_s)[order]
n = n_s[-1]
#file = open('output/measurements/first_optimization/%s_graph_measurements.txt'%n, "x")
with open('graph_measurements-Elkan-growing_dim.txt', 'w') as f:
    writer = csv.writer(f, delimiter='\t')
    writer.writerows(zip(n_s,sorted_graph))

with open('kmeans_measurements-Elkan-growing_dim.txt', 'w') as f:
    writer = csv.writer(f, delimiter='\t')
    writer.writerows(zip(n_s,sorted_kmeans))


"""
fp = open("perf_noeig_noturbo.txt", "r")
x = []
y = []
for line in fp:
    line = line.strip().split(" ")
    x.append(int(line[0]))
    y.append(float(line[1]))
"""

fig = plt.figure()
ax1 = fig.add_subplot(2,1,1)
ax1.plot(n_s, sorted_graph)

ax1.set(xlabel='d', ylabel='flops/cycle', title='Performance of Graph Construction number of dimensions, n= 2500, k = 5', ylim=0)

ax1.grid()

ax2 = fig.add_subplot(2,1,2)

ax2.plot(n_s, sorted_kmeans)

ax2.set(xlabel='d', ylabel='flops/cycle', title='Performance of Elkan-Kmeans number of dimensions, n= 2500, k = 5', ylim=0)

ax2.grid()
fig.tight_layout()
fig.savefig('graph_kmeans_growing_dim_Elkan.png')

#plt.show()



