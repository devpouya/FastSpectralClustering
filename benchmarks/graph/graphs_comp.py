import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import subprocess
import random
import os
import sys
import csv

input_path = os.getcwd() + "/output/measurements/graph_construction_speedup/100c/"
k=6

num_runs=9
dimensions = [2, 4,8, 16,32,64,128,256,512]

measurements = []

fig, ax = plt.subplots()

y= []

with open(input_path+"base.txt", 'r') as f:
    for line in f:
        line = line.strip().split("\t")
        print(line)
        y.append(int(line[1]))

    ax.plot(dimensions, y)

y= []

with open(input_path+"blocked.txt", 'r') as f:
    for line in f:
        line = line.strip().split("\t")
        print(line)
        y.append(int(line[1]))

    ax.plot(dimensions, y)

y= []

with open(input_path+"vec_8.txt", 'r') as f:
    for line in f:
        line = line.strip().split("\t")
        print(line)
        y.append(int(line[1]))

    ax.plot(dimensions, y)





ax.set(xlabel='dimension', ylabel='runtime [cycles]', title='Runtime of graphs constructions optimizations', ylim=0)
ax.grid()
plt.legend(['base', 'blocked','vec_8'], loc='upper left')
plt.xscale("log")
plt.xticks(dimensions,[str(dim) for dim in dimensions])
plt.show()
