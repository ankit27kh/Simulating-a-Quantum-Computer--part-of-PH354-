"""
Programming Project 2
Ankit Khandelwal
15863
"""

import numpy as np
from cmath import exp
from math import sqrt, pi
from random import random

N = 3 # Number of qubits
psi = np.ndarray((2**N,1), complex) # N-qubit register

H = 1/sqrt(2)*np.array([[1, 1],[1, -1]]) # Hadamard gate

def R(th):  # Phase shift gate
    return np.array([[1, 0],[0, exp(th*1j)]], complex)

I = np.eye(2, dtype=int) # Identity

# Circuits
U3a = np.kron(I, np.kron(H,I))
U3b = np.kron(H, np.kron(H,H))
U3c1 = np.kron(I, np.kron(I,H)) 
U3c2 = np.kron(I, np.kron(I,H)) 
U3d1 = np.kron(I, np.kron(I,H)) 
U3d2 = np.kron(I, np.kron(I,R(pi))) 
U3d3 = np.kron(I, np.kron(I,H)) 

repeat = 1000 # Repeat calculations 'repeat' times
result = np.zeros((repeat), int)
for ii in range(repeat):
    # psi is computational basis state
    comp_state = 0
    for i in range(2**N):
        if i == comp_state:
            psi[i] = 1
        else:
            psi[i] = 0
    if ii == 0:
        print('The Quantum State initially is:\n', psi)
    '''
    # for project 2 circuit (a)
    psi = U3a@psi
    '''
    '''
    # for project 2 circuit (b)
    psi = U3b@psi
    '''
    '''
    # for project 2 circuit (c)
    psi = U3c2@U3c1@psi
    '''
    
    # for project 2 circuit (d)
    psi = U3d3@U3d2@U3d1@psi
     
    
    #print('The Quantum State after computation is:\n', psi)
    # Measurment
    r = random()
    q = 0
    for i in range(2**N):
        q = q + abs(psi[i])**2
        if r < q:
            result[ii] = i  
            break
#print(result)
print('\nMeasurment Results:')
result_freq = np.zeros((2**N), int)
for i in range(2**N):
    result_freq[i] = np.count_nonzero(result == i)
    print(format(i, '0{}b'.format(N)), 'occured', result_freq[i], 'times.')
