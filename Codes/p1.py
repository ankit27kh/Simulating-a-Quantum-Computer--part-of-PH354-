"""
Programming Project 1
Ankit Khandelwal
15863
"""

import numpy as np
import cmath as c
from random import random

N = 3 # Number of qubits
psi = np.ndarray((2**N,1), complex) # N-qubit register

'''
# psi is computational basis state
comp_state = 5
for i in range(2**N):
    if i == comp_state:
        psi[i] = 1
    else:
        psi[i] = 0
'''
'''
# psi is equal superposition
psi[:] = 1/c.sqrt(2**N)
'''

# psi is cat state
for i in range(2**N):
    if i == 0 or i == 2**N-1:
        psi[i] = 1/c.sqrt(2)
    else:
        psi[i] = 0


#print('The Quantum State is:\n', psi)

# 'repeat' measurments of the Quantum state Psi
repeat = 10000
result = np.zeros((repeat), int)
for j in range(repeat):
    r = random()
    q = 0
    for i in range(2**N):
        q = q + abs(psi[i])**2
        if r < q:
            result[j] = i  
            break

#print(result)
result_freq = np.zeros((2**N), int)
for i in range(2**N):
    result_freq[i] = np.count_nonzero(result == i)
    print(format(i, '0{}b'.format(N)), 'occured', result_freq[i], 'times.', (result_freq[i]/repeat*100), 'percent.')
