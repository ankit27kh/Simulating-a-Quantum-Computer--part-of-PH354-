"""
Programming Project 3/5
Ankit Khandelwal
15863
"""

import numpy as np
from math import sqrt, pi
from random import random
from time import perf_counter

t_start = perf_counter()
N = 3 # Number of qubits
psi = np.ndarray((2**N,1), complex) # N-qubit register

H = 1/sqrt(2)*np.array([[1, 1],[1, -1]]) # Hadamard gate

I = np.eye(2, dtype=int) # Identity

oracle = np.eye(2**N, dtype=int)
ans = 7 # correct answer
print(format(ans, '0{}b'.format(N)), 'is correct.')
oracle[ans][ans] = -1 
#print(oracle)

Q = int(pi/4*sqrt(2**N))   # No. of iterations
print('Optimal iterations = {} for {} qubits.'.format(Q, N))

Q = Q + 0  # Change to see effect of iterations.

def J_0(N):
    J_0 = np.kron(H, H)
    for i in range(N-2):
        J_0 = np.kron(H, J_0)
    return J_0

J = np.eye(2**N, dtype=int)
J[0][0] = -1
gdo = J_0(N)@J@J_0(N) # Grover Diffusion Operator
t_stop = perf_counter()

print('\nTime taken = ', t_stop-t_start, 'seconds for N =', N, 'in calculating matrices.')
if N > 5:
    size = J.nbytes + oracle.nbytes + gdo.nbytes
    print('Memory usage for major matrices = ', size/1024/1024, 'MBytes.')

repeat = 10 # Repeat calculations 'repeat' times
result = np.zeros((repeat), int)
for j in range(repeat):
    # psi is computational basis state
    comp_state = 0
    for i in range(2**N):
        if i == comp_state:
            psi[i] = 1
        else:
            psi[i] = 0

    # step 1
    psi = J_0(N)@psi
    
    # step 2
    for i in range(Q):
        psi = gdo@oracle@psi
    
    # step 3
        r = random()
        q = 0
        for i in range(2**N):
            q = q + abs(psi[i])**2
            if r < q:
                result[j] = i  
                break

#print(result)
if N < 5:
    print('\nMeasurment Results:')
result_freq = np.zeros((2**N), int)
for i in range(2**N):
    result_freq[i] = np.count_nonzero(result == i)
    if N < 5:
        print(format(i, '0{}b'.format(N)), 'occured', result_freq[i], 'times.') 

correct_percent = result_freq[ans]/repeat*100
print('\nThe answer was correct', correct_percent, 'percent with {} iterations.'.format(Q))

t_stop = perf_counter()

print('\nTotal time taken = ', t_stop-t_start, 'seconds for N =', N, 'for {} repetitons.'.format(repeat))
