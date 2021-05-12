"""
Programming Project 10
Ankit Khandelwal
15863
"""

from math import sqrt, asin, sin, cos
import numpy as np
from cmath import exp
from random import random

N = 1

psi = np.ndarray((2**N,1), complex) # N-qubit register
comp_state = 0
for i in range(2**N):
    if i == comp_state:
        psi[i] = 1
    else:
        psi[i] = 0
psi_state_0 = np.copy(psi)

error_prob_percent = 10 
error = asin(sqrt(error_prob_percent/100))

H = 1/sqrt(2)*np.array([[1, 1],[1, -1]]) # Hadamard gate

# X Error
EX = np.array(([cos(error), -sin(error)*1j], [-sin(error)*1j, cos(error)]), complex)
# Z Error
EZ = np.array(([exp(-error*1j), 0], [0, exp(error*1j)]), complex)

X = np.array(([0, 1], [1, 0]), int) # X gate
Z = np.array(([1, 0], [0, -1]), int) # Z gate

repeat = 1000
result = np.ndarray((6, repeat), np.ndarray)

print('Error prob is {} and correct answer is 0.\n'.format(error_prob_percent))
for k in range(repeat):
    PSI = []
    psi = np.copy(psi_state_0)
    
    # for project 10 circuit (a)
    PSI.append(H@H@psi)
    psi = np.copy(psi_state_0)
    
    # for project 10 circuit (b)
    PSI.append(H@H@EX@psi)
    psi = np.copy(psi_state_0)
    
    # for project 10 circuit (c)
    PSI.append(H@H@X@psi)
    psi = np.copy(psi_state_0)
    
    # for project 10 circuit (d)
    PSI.append(H@EZ@H@psi)
    psi = np.copy(psi_state_0)
    
    # for project 10 circuit (e)
    PSI.append(H@Z@H@psi)
    psi = np.copy(psi_state_0)
    
    # for project 10 circuit (f)
    PSI.append(H@EZ@H@EX@psi)
    
    for n in range(6):
        r = random()
        q = 0
        for i in range(2**N):
            q = q + abs(PSI[n][i])**2
            if r < q:
                result[n][k] = i  
                break
fig = ['a', 'b', 'c', 'd', 'e', 'f']
for n in range(6):
    result_freq = np.zeros((2**N), int)
    print('For project 10 circuit({})'.format(fig[n]))
    for i in range(2**N):
        result_freq[i] = np.count_nonzero(result[n][:] == i)
        print(format(i, '0{}b'.format(N)), 'occured', result_freq[i], 'times.', (result_freq[i]/repeat*100), 'percent.')
    print('***')
