import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import subprocess
import random
import os
import sys
import csv

if (len(sys.argv) < 3):
    print("Format: (required)</path to dataset files/>, (required)<output file name>, (optional)<specify # clusters if growing n>")
    exit(0)

dataset_path = os.getcwd()+str(sys.argv[1])
output_file = sys.argv[2]
k = sys.argv[3]

output_path = os.getcwd() + "/output/measurements/"
print(dataset_path)

subprocess.run(["make", "countops"])
directory = os.fsencode(str(dataset_path))

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
        output = subprocess.check_output(["./countops", str(dataset_path)+f, str(k), "out.txt"],universal_newlines="\n").split("\n")
        num_cycle_graph = float(output[2])
        num_ops_graph = float(output[3])
        num_cycle_kmeans = float(output[4])
        num_ops_kmeans = float(output[5])

        #n = int(filename.split(".")[0])
        n_s.append(k)
        print("iteration %s\n"%i)
        graph_measurements.append(num_ops_graph/num_cycle_graph)
        kmeans_measurements.append(num_ops_kmeans/num_cycle_kmeans)
        i+=1

order = np.argsort(n_s)
sorted_graph = np.array(graph_measurements)[order]
sorted_kmeans = np.array(kmeans_measurements)[order]
n_s = np.array(n_s)[order]
n = n_s[-1]

with open(str(output_path)+"graph_"+str(output_file), 'w') as f:
    writer = csv.writer(f, delimiter='\t')
    writer.writerows(zip(n_s, sorted_graph))

with open(str(output_path)+"kmeans_"+str(output_file), 'w') as f:
    writer = csv.writer(f, delimiter='\t')
    writer.writerows(zip(n_s,sorted_kmeans))


fig = plt.figure()
ax1 = fig.add_subplot(2, 1, 1)
ax1.plot(n_s, sorted_graph)

ax1.set(xlabel='d', ylabel='flops/cycle', title='', ylim=0)
ax1.grid()
ax2 = fig.add_subplot(2, 1, 2)
ax2.plot(n_s, sorted_kmeans)
ax2.set(xlabel='d', ylabel='flops/cycle', title='', ylim=0)

ax2.grid()
fig.tight_layout()
# fig.savefig('graph_.png')

plt.show()



