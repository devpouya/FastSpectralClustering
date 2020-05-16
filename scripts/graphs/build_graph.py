import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import subprocess
import random
import os
import sys
import csv

output_path = os.getcwd()+"/output/measurements/";

with open(output_path+"graph_base_growk.txt", 'r') as f:
    lines = f.readlines()
    n_s = [float(line.split()[0]) for line in lines]
    sorted_graph = [float(line.split()[1]) for line in lines]
  #  print(sorted_graph)

with open(output_path+"kmeans_base_growk.txt", 'r') as f:
    lines = f.readlines()
    n_s = [float(line.split()[0]) for line in lines]
    sorted_kmeans = [float(line.split()[1]) for line in lines]


fig = plt.figure()
ax1 = fig.add_subplot(2, 1, 1)
ax1.plot(n_s, sorted_graph)

ax1.set(xlabel='k (number of clusters)', ylabel='flops/cycle', title='Base l2_norm on Graph construction', ylim=0)
ax1.grid()
ax2 = fig.add_subplot(2, 1, 2)
ax2.plot(n_s, sorted_kmeans)
ax2.set(xlabel='k (number of clusters)', ylabel='flops/cycle', title='Base l2_norm on Hamerly', ylim=0)

ax2.grid()
fig.tight_layout()
# fig.savefig('graph_.png')

plt.show()
