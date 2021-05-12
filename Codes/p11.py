"""
Programming Project 11
Ankit Khandelwal
15863
"""

from math import sqrt, sin, cos, asin
import numpy as np
from cmath import exp
from random import random

N = 2
measure_only = 1
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

EX = np.array(([cos(error), -1j*sin(error)], [-1j*sin(error), cos(error)]), complex)
EZ = np.array(([exp(-1j*error), 0], [0, exp(1j*error)]), complex)
X = np.array(([0, 1], [1, 0]), int)
Z = np.array(([1, 0], [0, -1]), int)

def kd(i,j):
    if i == j:
        return 1
    else:
        return 0

C_NOT = np.zeros((4, 4), int)
C_NOT[0][0] = 1
C_NOT[1][1] = 1
C_NOT[2][3] = 1
C_NOT[3][2] = 1

def C_NOTn(a,b):
    C_NOTn = np.ndarray((2**N, 2**N))
    bits = 2**N 
    extra = []
    for r in range(0, N):
        extra.append(int(r))
    extra.remove(a-1)
    extra.remove(b-1)
    for i in range(bits):
        for j in range(bits):
            final = 1
            ii = str(format(i, '0{}b'.format(N))) 
            jj = str(format(j, '0{}b'.format(N))) 
            index1 = np.zeros((N), str)
            index2 = np.zeros((N), str)
            for k in range(N):
                index1[k] = ii[k]
                index2[k] = jj[k]
            #print(index1, index2)
            qp = int(str(index1[a-1] + index1[b-1]), 2)
            q_p_ = int(str(index2[a-1] + index2[b-1]), 2)
            for m in extra:
                final = final * kd(index1[m], index2[m])
            #print(final)
            C_NOTn[i][j] = C_NOT[qp][q_p_]*final
    return C_NOTn

def multi_gate(gate, n):
    right = N-n
    left = N-right-1
    if right > 0:
        RIGHT = np.eye(2**right, dtype=int)
    if left > 0:
        LEFT = np.eye(2**left, dtype=int)
    if right == 0:
        return np.kron(LEFT, gate)
    if left == 0:
        return np.kron(gate, RIGHT)
    return (np.kron(LEFT, np.kron(gate, RIGHT)))

repeat = 1000
result = np.ndarray((2, repeat), np.ndarray)

for k in range(repeat):
    PSI = []
    psi = np.copy(psi_state_0)
    
    # for project 11 circuit (a)
    PSI.append(multi_gate(H,1)@multi_gate(H,1)@C_NOTn(2,1)@multi_gate(EX,2)@psi)
    psi = np.copy(psi_state_0)
    
    # for project 11 circuit (b)
    PSI.append(multi_gate(H,1)@C_NOTn(1,2)@multi_gate(H,1)@psi)
        
    for n in range(2):
        r = random()
        q = 0
        for i in range(2**N):
            q = q + abs(PSI[n][i])**2
            if r < q:
                result[n][k] = (i)  
                break
for n in range(2):
    if n == 0:print('Project 11 circuit (a)')
    else: print('Project 11 circuit (b)') 
    result_freq = np.zeros((2**N), int)
    for i in range(2**N):
        result_freq[i] = np.count_nonzero(result[n][:] == i)
        print(format(i, '0{}b'.format(N)), 'occured', result_freq[i], 'times.', (result_freq[i]/repeat*100), 'percent.')
    print('***')
    if measure_only == N:
        continue
    modified_result_freq = np.zeros((2**measure_only), int)
    for ii in range(2**measure_only):
        for i in range(2**(N-measure_only)):
            modified_result_freq[ii] = modified_result_freq[ii] + result_freq[i + ii*(2**(N-measure_only))] 
        print(format(ii, '0{}b'.format(measure_only)), 'occured', modified_result_freq[ii], 'times.', (modified_result_freq[ii]/repeat*100), 'percent.')
    print('++++++++\n')
