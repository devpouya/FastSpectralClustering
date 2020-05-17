import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import subprocess
import random
import os
import sys
import csv

input_path = os.getcwd() + "/output/measurements/blocksize_growingdim_median/"
k=6

num_runs=9
dimensions = [2, 4,8, 16,32,64,128,256,512]
blocksizes = [4,8, 16,32,64,128,256]
measurements = []

fig, ax = plt.subplots()

for bs in blocksizes:
    print("Dimension: "+str(bs))
    filename = str(bs)+".txt"
    y = []
    with open(input_path+filename, 'r') as f:
        for line in f:
            line = line.strip().split("\t")
            print(line)
            y.append(int(line[1]))

        ax.plot(dimensions, y)





ax.set(xlabel='dimension', ylabel='runtime [cycles]', title='Effect of blocksize in blocked graph construction', ylim=0)
ax.grid()
plt.legend(['bs = 4', 'bs = 8', 'bs = 16', 'bs = 32','bs = 64','bs = 128','bs = 256'], loc='upper left')
plt.xscale("log")
plt.xticks(dimensions,[str(dim) for dim in dimensions])
plt.show()
