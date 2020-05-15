import matplotlib
import matplotlib.pyplot as plt
import sys
import os

if (len(sys.argv) < 3):
    print("Format: <number of lines> [</performances file with path from root>] (at least 1) (change legend in python)")
    exit(0)

n = sys.argv[1]
perf_path = []

for i in range(0, n):
    perf_path[i] = os.getcwd() + str(sys.argv[i])

    fp = open(perf_path[i], "r")
    x = []
    y = []
    for line in fp:
        line = line.strip().split(" ")
        x.append(int(line[0]))
        y.append(float(line[1]))

    fig, ax = plt.subplots()

    ax.plot(x, y)

ax.set(xlabel='n', ylabel='flops/cycle', title='', ylim=0)
ax.grid()
fig.tight_layout()
plt.show()
