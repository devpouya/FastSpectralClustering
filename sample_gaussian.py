import numpy as np
mu0, sigma0 = 0.0, 0.1
mu1, sigma1 = 1.0, 0.1

s0 = np.random.normal(mu0, sigma0, 50)
print(s0[0:10,])
print(s0.shape)

s1 = np.random.normal(mu1, sigma1, 50)
print(s1[0:10,])
print(s1.shape)

s = np.concatenate((s0,s1))
np.random.shuffle(s)
print(s[0:10,])
print(s.shape)

file = open("./gaussian_1d_k2.txt","w+") 
file.write("1\n")
for i in range(100):
    file.write(str(s[i]) + "\n")

#################################
mean0 = [-2,-2]
mean1 = [2,  2]
cov = [[1, 0], [0, 1]]
g0 = np.random.multivariate_normal(mean0, cov, 50)
print(g0[0:5,])
g1 = np.random.multivariate_normal(mean1, cov, 50)
print(g1[0:5,])
g = np.concatenate((g0,g1))
np.random.shuffle(g)
print(g.shape)
print(g[0:5,])
file = open("./gaussian_2d_k2.txt","w+") 
file.write("2\n")
for i in range(100):
    for j in range(2):
        if j != 1:
            file.write(str(g[i][j]) + " ")
        else:
            file.write(str(g[i][j]) + "\n")

            










