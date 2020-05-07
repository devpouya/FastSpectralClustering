import matplotlib
import matplotlib.pyplot as plt

fp = open("perf_noeig_noturbo.txt", "r")
x = []
y = []
for line in fp:
    line = line.strip().split(" ")
    x.append(int(line[0]))
    y.append(float(line[1]))

fig, ax = plt.subplots()

ax.plot(x, y)

ax.set(xlabel='n', ylabel='flops/cycle', title='Performance of spectral base_clustering (2 clusters, $n$ = number of data points)', ylim=0)

ax.grid()
fig.tight_layout()
plt.show()
