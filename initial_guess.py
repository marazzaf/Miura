#coding: utf-8

import numpy as np

#define values
theta = np.pi/2
L = 2*np.sin(0.5*np.arccos(0.5/np.cos(0.5*theta))) #length of rectangle
alpha = np.sqrt(1 / (1 - np.sin(theta/2)**2))
H = 2*np.pi/alpha #height of rectangle
l = sin(theta/2)*L #total height of cylindre
alpha = 0.02 #variation at the top


#writing the matrix of the system
A = np.zeros(6,6)
#Filling-in by line
A[0,:] = np.array([])  
