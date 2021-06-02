"""
Programming Project 4
Ankit Khandelwal
15863
"""

from math import sqrt
from random import random
from time import perf_counter

import numpy as np

t_start = perf_counter()


# Kronecker delta
def kd(i, j):
    if i == j:
        return 1
    else:
        return 0


# CNOT gate
C_NOT = np.zeros((4, 4), int)
C_NOT[0][0] = 1
C_NOT[1][1] = 1
C_NOT[2][3] = 1
C_NOT[3][2] = 1

N = 3  # Number of Qubits


# CNOT gate for N qubits with control qubit 'a' and target qubit 'b'
def C_NOTn(a, b):
    C_NOTn = np.ndarray((2 ** N, 2 ** N))
    bits = 2 ** N
    extra = []
    for r in range(0, N):
        extra.append(int(r))
    extra.remove(a - 1)
    extra.remove(b - 1)
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
            qp = int(str(index1[a - 1] + index1[b - 1]), 2)
            q_p_ = int(str(index2[a - 1] + index2[b - 1]), 2)
            for m in extra:
                final = final * kd(index1[m], index2[m])
            C_NOTn[i][j] = C_NOT[qp][q_p_] * final
    return C_NOTn


H = 1 / sqrt(2) * np.array([[1, 1], [1, -1]])  # Hadamard gate

I = np.eye(2, dtype=int)  # Identity

psi = np.ndarray((2 ** N, 1), complex)  # N-qubit register

# Circuits
U6a1 = np.kron(I, np.kron(H, I))
U6a2 = C_NOTn(2, 3)
U6b1 = U6a1
U6b2 = U6a2
U6b3 = C_NOTn(2, 1)
U6c1 = U6a1
U6c2 = U6a1
U6d1 = U6a1
U6d2 = U6a2
U6d3 = U6a1

repeat = 1000  # Repeat calculations 'repeat' times
result = np.zeros((repeat), int)
for ii in range(repeat):
    # psi is computational basis state
    comp_state = 0
    for i in range(2 ** N):
        if i == comp_state:
            psi[i] = 1
        else:
            psi[i] = 0
    '''
    # for project 4 circuit (a)
    psi = U6a2@U6a1@psi
    '''
    '''
    # for project 4 circuit (b)
    psi = U6b3@U6b2@U6b1@psi
    '''
    '''
    # for project 4 circuit (c)
    psi = U6c2@U6c1@psi
    '''

    # for project 4 circuit (d)
    psi = U6d3 @ U6d2 @ U6d1 @ psi

    # Measurment
    rand = random()
    q = 0
    for i in range(2 ** N):
        q = q + abs(psi[i]) ** 2
        if rand < q:
            result[ii] = i
            break
# print(result)
print('\nMeasurment Results:')
result_freq = np.zeros((2 ** N), int)
for i in range(2 ** N):
    result_freq[i] = np.count_nonzero(result == i)
    print(format(i, '0{}b'.format(N)), 'occured', result_freq[i], 'times.')
