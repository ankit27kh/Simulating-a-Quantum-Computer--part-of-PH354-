"""
Programming Project 14
Ankit Khandelwal
15863
"""

import cmath
from cmath import sin, cos, exp
from math import sqrt, asin
from random import choices

import numpy as np
import scipy.sparse as sp

'''
At all 3 error locations in the code, it can correct 1 bit flip error.
The QEC block can be repeated after every step in a calulation to correct single bit flip errors.
'''

N = 5
psi = np.ndarray((2 ** N, 1), complex)  # N-qubit register
comp_state = 0
for i in range(2 ** N):
    if i == comp_state:
        psi[i] = 1
    else:
        psi[i] = 0

j = cmath.sqrt(-1)
error_prob_percent = 10
error = asin(sqrt(error_prob_percent / 100))

H = 1 / sqrt(2) * np.array([[1, 1], [1, -1]])  # Hadamard gate

EX = np.array(([cos(error), -j * sin(error)], [-j * sin(error), cos(error)]), complex)
EZ = np.array(([exp(-j * error), 0], [0, exp(j * error)]), complex)
X = np.array(([0, 1], [1, 0]), int)
Z = np.array(([1, 0], [0, -1]), int)


def kd(i, j):
    if i == j:
        return 1
    else:
        return 0


zero_ket = np.ndarray((2, 1), int)
zero_ket[0][0] = 1
zero_ket[1][0] = 0
zero_bra = zero_ket.T

one_ket = np.ndarray((2, 1), int)
one_ket[0][0] = 0
one_ket[1][0] = 1
one_bra = one_ket.T

zero_matrix = zero_ket @ zero_bra
one_matrix = one_ket @ one_bra


def C_NOTn(a, b):
    # part 1
    left = a - 1
    right = N - a
    if right > 0:
        RIGHT = sp.eye(2 ** right, dtype=int)
    if left > 0:
        LEFT = sp.eye(2 ** left, dtype=int)
    if left == 0:
        LEFT = 1
    if right == 0:
        RIGHT = 1
    part1 = sp.kron(LEFT, sp.kron(zero_matrix, RIGHT))
    # part2
    if a < b:
        a = a
        b = b
        count = 1
    else:
        temp = a
        a = b
        b = temp
        count = 2
    left = a - 1
    right = N - b
    center = N - left - right - 2
    if right > 0:
        RIGHT = sp.eye(2 ** right, dtype=int)
    if left > 0:
        LEFT = sp.eye(2 ** left, dtype=int)
    if left == 0:
        LEFT = 1
    if right == 0:
        RIGHT = 1
    if center > 0:
        CENTER = sp.eye(2 ** center, dtype=int)
    if center == 0:
        CENTER = 1
    if count == 1:
        part2 = sp.kron(LEFT, sp.kron(one_matrix, sp.kron(CENTER, sp.kron(X, RIGHT))))
    if count == 2:
        part2 = sp.kron(LEFT, sp.kron(X, sp.kron(CENTER, sp.kron(one_matrix, RIGHT))))
    return part1 + part2


def multi_gate(gate, n):
    right = N - n
    left = N - right - 1
    if right > 0:
        RIGHT = sp.eye(2 ** right, dtype=int)
    if left > 0:
        LEFT = sp.eye(2 ** left, dtype=int)
    if right == 0:
        return sp.kron(LEFT, gate)
    if left == 0:
        return sp.kron(gate, RIGHT)
    return sp.csr_matrix((sp.kron(LEFT, sp.kron(gate, RIGHT))))


def P_0(n):
    P0 = np.zeros((2 ** N, 2 ** N), int)
    for k in range(2 ** N):
        jj = str(format(k, '0{}b'.format(N)))
        if jj[N - n] == '0':
            P0[k][k] = kd(k, k)
        else:
            P0[k][k] = 0
    return (P0)


measure_qubits = [1, 2, 3, 4, 5]  # All Qubits

PP_0 = np.ndarray((N + 1), np.ndarray)
for n in measure_qubits:
    PP_0[n] = (P_0(n))


def P_1(n):
    P1 = np.zeros((2 ** N, 2 ** N), int)
    for k in range(2 ** N):
        P1[k][k] = kd(k, k) - PP_0[n][k][k]
    return sp.csr_matrix(P1)


PP_1 = np.ndarray((N + 1), np.ndarray)

for n in measure_qubits:
    PP_1[n] = (P_1(n))
    PP_0[n] = sp.csr_matrix(PP_0[n])
prob = np.zeros((N + 1, 2), float)

repeat = 100
results = np.ndarray((repeat, N), int)
for ii in range(repeat):
    # psi is computational basis state
    comp_state = 0
    for i in range(2 ** N):
        if i == comp_state:
            psi[i] = 1
        else:
            psi[i] = 0

    psi = multi_gate(X, 2) @ psi  # Error

    ######## QUANTUM ERROR CORRECTION #########
    measure_qubits = [2, 1]  # Needs to be custom in each code in (N+1-i) format
    # where i is the qubit to be measured

    psi = C_NOTn(3, 5) @ C_NOTn(1, 5) @ C_NOTn(2, 4) @ C_NOTn(1, 4) @ psi
    ## MEASURMENT BLOCK
    for n in measure_qubits:
        prob[n][0] = (((PP_0[n] @ psi).conj().T) @ (PP_0[n] @ psi)).real
        prob[n][1] = (((PP_1[n] @ psi).conj().T) @ (PP_1[n] @ psi)).real
        r = choices((0, 1), (prob[n][0], prob[n][1]))
        r = r[0]
        # print(r)
        # print(psi)
        if r == 0:
            psi = (PP_0[n] @ psi) / sqrt(prob[n][0])
        else:
            psi = (PP_1[n] @ psi) / sqrt(prob[n][1])
        results[ii][N - n] = r

    S1 = results[ii][3]
    S2 = results[ii][4]
    # print(S1,S2)
    if S1 == 0 and S2 == 0:
        psi = psi
    if S1 == 0 and S2 == 1:
        psi = multi_gate(X, 3) @ psi
        psi = multi_gate(X, 5) @ psi
    if S1 == 1 and S2 == 0:
        psi = multi_gate(X, 2) @ psi
        psi = multi_gate(X, 4) @ psi
    if S1 == 1 and S2 == 1:
        psi = multi_gate(X, 1) @ psi
        psi = multi_gate(X, 4) @ multi_gate(X, 5) @ psi
    ################################################

    psi = multi_gate(X, 2) @ psi  # Error

    ######## QUANTUM ERROR CORRECTION #########
    measure_qubits = [2, 1]  # Needs to be custom in each code in (N+1-i) format
    # where i is the qubit to be measured

    psi = C_NOTn(3, 5) @ C_NOTn(1, 5) @ C_NOTn(2, 4) @ C_NOTn(1, 4) @ psi
    ## MEASURMENT BLOCK
    for n in measure_qubits:
        prob[n][0] = (((PP_0[n] @ psi).conj().T) @ (PP_0[n] @ psi)).real
        prob[n][1] = (((PP_1[n] @ psi).conj().T) @ (PP_1[n] @ psi)).real
        r = choices((0, 1), (prob[n][0], prob[n][1]))
        r = r[0]
        # print(r)
        # print(psi)
        if r == 0:
            psi = (PP_0[n] @ psi) / sqrt(prob[n][0])
        else:
            psi = (PP_1[n] @ psi) / sqrt(prob[n][1])
        results[ii][N - n] = r

    S1 = results[ii][3]
    S2 = results[ii][4]
    # print(S1,S2)
    if S1 == 0 and S2 == 0:
        psi = psi
    if S1 == 0 and S2 == 1:
        psi = multi_gate(X, 3) @ psi
        psi = multi_gate(X, 5) @ psi
    if S1 == 1 and S2 == 0:
        psi = multi_gate(X, 2) @ psi
        psi = multi_gate(X, 4) @ psi
    if S1 == 1 and S2 == 1:
        psi = multi_gate(X, 1) @ psi
        psi = multi_gate(X, 4) @ multi_gate(X, 5) @ psi
    ################################################

    psi = multi_gate(X, 1) @ psi  # Error

    measure_qubits = [3, 4, 5]
    ## MEASURMENT BLOCK
    for n in measure_qubits:
        prob[n][0] = (((PP_0[n] @ psi).conj().T) @ (PP_0[n] @ psi)).real
        prob[n][1] = (((PP_1[n] @ psi).conj().T) @ (PP_1[n] @ psi)).real
        r = choices((0, 1), (prob[n][0], prob[n][1]))
        r = r[0]
        # print(r)
        # print(psi)
        if r == 0:
            psi = (PP_0[n] @ psi) / sqrt(prob[n][0])
        else:
            psi = (PP_1[n] @ psi) / sqrt(prob[n][1])
        results[ii][N - n] = r
    a = np.zeros((3))
    for i in range(3):
        a[i] = results[ii][i]
    if a[0] + a[1] + a[2] < 2:
        for i in range(3):
            results[ii][i] = 0
    else:
        for i in range(3):
            results[ii][i] = 1

results_list = []
for i in range(repeat):
    strr = ''
    for j in range(N - 2):
        strr = strr + str(results[i][j])
    results_list.append(strr)
# print(result)
print('\nMeasurment Results:')
result_freq = np.zeros((2 ** (N - 2)), int)
for i in range(2 ** (N - 2)):
    ii = str(format(i, '0{}b'.format(N - 2)))
    result_freq[i] = results_list.count(ii)
    print(format(i, '0{}b'.format(N - 2)), 'occured', result_freq[i], 'times.')
###############################################################################
