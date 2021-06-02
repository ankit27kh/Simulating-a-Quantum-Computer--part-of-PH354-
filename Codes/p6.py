"""
Programming Project 6
Ankit Khandelwal
15863
"""

from math import sqrt, pi
from random import random
from time import perf_counter

import numpy as np
from scipy import sparse as sp

t_start = perf_counter()
N = 10  # Number of qubits
psi = np.ndarray((2 ** N, 1), complex)  # N-qubit register

H = 1 / sqrt(2) * np.array([[1, 1], [1, -1]])  # Hadamard gate

oracle_diag = np.ones(2 ** N, dtype=int)
ans = 4  # correct answer
print(format(ans, '0{}b'.format(N)), 'is correct.')
oracle_diag[ans] = -1
oracle = sp.csr_matrix(sp.diags(oracle_diag, dtype=int))
# print(oracle)

Q = int(pi / 4 * sqrt(2 ** N))  # No. of iterations
print('Optimal iterations = {} for {} qubits.'.format(Q, N))
J_diag = np.ones(2 ** N, dtype=int)
J_diag[0] = -1
J = sp.csr_matrix(sp.diags(J_diag, dtype=int))

size = J.data.nbytes + J.indptr.nbytes + J.indices.nbytes
size = size + oracle.data.nbytes + oracle.indptr.nbytes + oracle.indices.nbytes
print('Memory usage for major matrices = ', size / 1024 / 1024, 'MBytes.')


# Hadamard Gate on qubit 'n' in N qubit circuit
def HH(n):
    right = N - n
    left = N - right - 1
    if right > 0:
        RIGHT = sp.eye(2 ** right, dtype=int)
    if left > 0:
        LEFT = sp.eye(2 ** left, dtype=int)
    if right == 0:
        return sp.kron(LEFT, H)
    if left == 0:
        return sp.kron(H, RIGHT)
    return sp.csr_matrix(sp.kron(LEFT, sp.kron(H, RIGHT)))


repeat = 1  # Repeat calculations 'repeat' times
result = np.zeros((repeat), int)
for qq in range(repeat):
    comp_state = 0
    for i in range(2 ** N):
        if i == comp_state:
            psi[i] = 1
        else:
            psi[i] = 0
    # psi is computational basis state

    for i in range(N):
        psi = HH(i + 1) @ psi

    for i in range(Q):
        psi = oracle @ psi
        for j in range(N):
            psi = HH(j + 1) @ psi
        psi = J @ psi
        for j in range(N):
            psi = HH(j + 1) @ psi
    # print(psi)
    r = random()
    q = 0
    for i in range(2 ** N):
        q = q + abs(psi[i]) ** 2
        if r < q:
            result[qq] = i
            break

result_freq = np.zeros((2 ** N), int)
for i in range(2 ** N):
    result_freq[i] = np.count_nonzero(result == i)

correct_percent = result_freq[ans] / repeat * 100
print('\nThe answer was correct', correct_percent, 'percent with {} iterations.'.format(Q))
t_stop = perf_counter()
print('\nTotal time taken = ', t_stop - t_start, 'seconds for N =', N, 'for {} repetitons.'.format(repeat))
