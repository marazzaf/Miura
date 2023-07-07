#coding: utf-8

import numpy as np

data = np.loadtxt('rate.txt', delimiter = ',')

a = data[:,0]
b = data[:,1]

res = 2*np.log(b[:-1]/b[1:]) / np.log(a[1:]/a[:-1])
print(res)
