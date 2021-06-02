"""
Programming Project 15
Ankit Khandelwal
15863
"""

from math import sqrt
from random import choices, randint

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp

N = 15
xnos = 3
znos = 4
dnos = 8
data_qubits = np.ndarray((dnos, 2), list)
data_qubits[0][0] = 1
data_qubits[1][0] = 3
data_qubits[2][0] = 5
data_qubits[3][0] = 7
data_qubits[4][0] = 9
data_qubits[5][0] = 11
data_qubits[6][0] = 13
data_qubits[7][0] = 15
data_qubits[0][1] = [1, 3]
data_qubits[1][1] = [3, 3]
data_qubits[2][1] = [5, 3]
data_qubits[3][1] = [2, 2]
data_qubits[4][1] = [4, 2]
data_qubits[5][1] = [1, 1]
data_qubits[6][1] = [3, 1]
data_qubits[7][1] = [5, 1]

X_stabilizers = np.ndarray((xnos, 6), list)
X_stabilizers[0][0] = 6
X_stabilizers[0][1] = 1
X_stabilizers[0][2] = 'a'
X_stabilizers[0][3] = 7
X_stabilizers[0][4] = 11
X_stabilizers[0][5] = [1, 2]
X_stabilizers[1][0] = 8
X_stabilizers[1][1] = 3
X_stabilizers[1][2] = 7
X_stabilizers[1][3] = 9
X_stabilizers[1][4] = 13
X_stabilizers[1][5] = [3, 2]
X_stabilizers[2][0] = 10
X_stabilizers[2][1] = 5
X_stabilizers[2][2] = 9
X_stabilizers[2][3] = 'a'
X_stabilizers[2][4] = 15
X_stabilizers[2][5] = [5, 2]

Z_stabilizers = np.ndarray((znos, 6), list)
Z_stabilizers[0][0] = 2
Z_stabilizers[0][1] = 'a'
Z_stabilizers[0][2] = 1
Z_stabilizers[0][3] = 3
Z_stabilizers[0][4] = 7
Z_stabilizers[0][5] = [2, 3]
Z_stabilizers[1][0] = 4
Z_stabilizers[1][1] = 'a'
Z_stabilizers[1][2] = 3
Z_stabilizers[1][3] = 5
Z_stabilizers[1][4] = 9
Z_stabilizers[1][5] = [4, 3]
Z_stabilizers[2][0] = 12
Z_stabilizers[2][1] = 7
Z_stabilizers[2][2] = 11
Z_stabilizers[2][3] = 13
Z_stabilizers[2][4] = 'a'
Z_stabilizers[2][5] = [2, 1]
Z_stabilizers[3][0] = 14
Z_stabilizers[3][1] = 9
Z_stabilizers[3][2] = 13
Z_stabilizers[3][3] = 15
Z_stabilizers[3][4] = 'a'
Z_stabilizers[3][5] = [4, 1]

for n in range(dnos):
    d = plt.scatter(data_qubits[n][1][0], data_qubits[n][1][1], c='r', s=1000)
for n in range(xnos):
    x = plt.scatter(X_stabilizers[n][5][0], X_stabilizers[n][5][1], c='b', s=1000)
for n in range(znos):
    z = plt.scatter(Z_stabilizers[n][5][0], Z_stabilizers[n][5][1], c='g', s=1000)
plt.title('Surface Code Computer')
plt.xticks([1, 2, 3, 4, 5])
plt.yticks([1, 2, 3])
plt.legend((d, x, z), ('Data Qubits', 'X-Stabilizers', 'Z-Stabilizers'), fontsize=12, scatterpoints=1, loc=(1.04, 0),
           markerscale=.4)
plt.show()

error_table = np.zeros((4 ** dnos, 1 + xnos + znos + 1), int)


def numToBase(n, b):
    digits = np.zeros(dnos, int)
    if n == 0:
        return digits
    i = 0
    while n:
        digits[i] = (int(n % b))
        n //= b
        i = i + 1
    return digits[::-1]


for i in range(4 ** dnos):
    temp = numToBase(i, 4)
    error_table[i][-2] = i
    error_table[i][-1] = np.count_nonzero(temp == 0)
    for k in range(dnos):
        if temp[k] == 1 or temp[k] == 3:
            qubit_loc = np.argwhere((Z_stabilizers[:, 1:-1]) == (data_qubits[k][0]))
            for l in range(len(qubit_loc)):
                error_table[i][qubit_loc[l][0] + xnos] = 1 - error_table[i][qubit_loc[l][0] + xnos]
        if temp[k] == 2 or temp[k] == 3:
            qubit_loc = np.argwhere((X_stabilizers[:, 1:-1]) == (data_qubits[k][0]))
            for l in range(len(qubit_loc)):
                error_table[i][qubit_loc[l][0]] = 1 - error_table[i][qubit_loc[l][0]]
error_table = error_table[error_table[:, -1].argsort()[::-1]]

psi = np.ndarray((2 ** N, 1), complex)  # N-qubit register
H = 1 / sqrt(2) * np.array([[1, 1], [1, -1]])  # Hadamard gate

X = np.array(([0, 1], [1, 0]), int)
Z = np.array(([1, 0], [0, -1]), int)
Y = X @ Z * 1j


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
    P0 = np.ndarray((1, 2 ** N), int)
    for k in range(2 ** N):
        jj = str(format(k, '0{}b'.format(N)))
        if jj[N - n] == '0':
            P0[0][k] = 1
        else:
            P0[0][k] = 0
    return P0


measure_qubits = np.concatenate(((N + 1) - X_stabilizers[:, 0], (N + 1) - \
                                 Z_stabilizers[:, 0])).reshape(1, xnos + znos)
# Needs to be custom in each code in (N+1-i) format
# where i is the qubit to be measured
PP_0 = np.ndarray((N + 1), np.ndarray)
for nn in measure_qubits:
    for n in nn:
        PP_0[n] = (P_0(n))


def P_1(n):
    P1 = np.ndarray((1, 2 ** N), int)
    for k in range(2 ** N):
        P1[0][k] = kd(k, k) - PP_0[n][0][k]
    return sp.csr_matrix(sp.diags(P1[0, :], 0))


PP_1 = np.ndarray((N + 1), np.ndarray)
for nn in measure_qubits:
    for n in nn:
        PP_1[n] = (P_1(n))
        PP_0[n] = sp.csr_matrix(sp.diags(PP_0[n][0, :], 0))
prob = np.zeros((N + 1, 2), float)
# psi is computational basis state
comp_state = randint(1, 2 ** N)
for i in range(2 ** N):
    if i == comp_state:
        psi[i] = 1
    else:
        psi[i] = 0


def error_type(change):
    index = error_table[:, 0:7].tolist().index(change.tolist())
    gate_list = numToBase(error_table[index][7], 4)
    for i in range(len(gate_list)):
        if gate_list[i] == 0:
            continue
        elif gate_list[i] == 1:
            print('X error on {} qubit.'.format(data_qubits[i][0]))
        elif gate_list[i] == 2:
            print('Z error on {} qubit.'.format(data_qubits[i][0]))
        elif gate_list[i] == 3:
            print('Y error on {} qubit.'.format(data_qubits[i][0]))


repeat = 10
results = 5 * np.ones((repeat, N), int)
syndrome_old = np.zeros(xnos + znos, int)
for ii in range(repeat):
    syndrome_new = np.zeros(xnos + znos, int)

    if ii == 1:  # Add desired error
        psi = multi_gate(X, 15) @ psi
    # psi = multi_gate(Y, 1)@psi

    # 1 Initialize
    for nn in measure_qubits:
        for n in nn:
            prob[n][0] = (((PP_0[n] @ psi).conj().T) @ (PP_0[n] @ psi)).real
            prob[n][1] = (((PP_1[n] @ psi).conj().T) @ (PP_1[n] @ psi)).real
            r = choices((0, 1), (prob[n][0], prob[n][1]))
            r = r[0]
            if r == 0:
                psi = (PP_0[n] @ psi) / sqrt(prob[n][0])
            else:
                psi = (PP_1[n] @ psi) / sqrt(prob[n][1])
            results[ii][N - n] = r

    syndome_qubits = np.concatenate((X_stabilizers[:, 0], \
                                     Z_stabilizers[:, 0])).reshape(1, xnos + znos)
    for nn in syndome_qubits:
        for n in nn:
            if results[ii][n - 1] == 1:
                psi = multi_gate(X, n) @ psi

    # 2 H
    for i in range(xnos):
        psi = multi_gate(H, X_stabilizers[i][0]) @ psi

    # 3 Up
    for i in range(xnos):
        if X_stabilizers[i][1] == 'a':
            continue
        psi = C_NOTn(X_stabilizers[i][0], X_stabilizers[i][1]) @ psi

    for i in range(znos):
        if Z_stabilizers[i][1] == 'a':
            continue
        psi = C_NOTn(Z_stabilizers[i][1], Z_stabilizers[i][0]) @ psi

    # 4 Left
    for i in range(xnos):
        if X_stabilizers[i][2] == 'a':
            continue
        psi = C_NOTn(X_stabilizers[i][0], X_stabilizers[i][2]) @ psi

    for i in range(znos):
        if Z_stabilizers[i][2] == 'a':
            continue
        psi = C_NOTn(Z_stabilizers[i][2], Z_stabilizers[i][0]) @ psi

    # 5 Right
    for i in range(xnos):
        if X_stabilizers[i][3] == 'a':
            continue
        psi = C_NOTn(X_stabilizers[i][0], X_stabilizers[i][3]) @ psi

    for i in range(znos):
        if Z_stabilizers[i][3] == 'a':
            continue
        psi = C_NOTn(Z_stabilizers[i][3], Z_stabilizers[i][0]) @ psi

    # 6 Down
    for i in range(xnos):
        if X_stabilizers[i][4] == 'a':
            continue
        psi = C_NOTn(X_stabilizers[i][0], X_stabilizers[i][4]) @ psi

    for i in range(znos):
        if Z_stabilizers[i][4] == 'a':
            continue
        psi = C_NOTn(Z_stabilizers[i][4], Z_stabilizers[i][0]) @ psi

    # 7 H
    for i in range(xnos):
        psi = multi_gate(H, X_stabilizers[i][0]) @ psi

    # 8 Measure
    for nn in measure_qubits:
        for n in nn:
            prob[n][0] = (((PP_0[n] @ psi).conj().T) @ (PP_0[n] @ psi)).real
            prob[n][1] = (((PP_1[n] @ psi).conj().T) @ (PP_1[n] @ psi)).real
            r = choices((0, 1), (prob[n][0], prob[n][1]))
            r = r[0]
            if r == 0:
                psi = (PP_0[n] @ psi) / sqrt(prob[n][0])
            else:
                psi = (PP_1[n] @ psi) / sqrt(prob[n][1])
            results[ii][N - n] = r

    for nn in syndome_qubits:
        i = 0
        for n in nn:
            syndrome_new[i] = (results[ii][n - 1])
            i = i + 1
    # print(syndrome_new)
    if ii == 0:
        syndrome_old = np.copy(syndrome_new)
    if ii != 0:
        change = abs(syndrome_new - syndrome_old)
        syndrome_old = np.copy(syndrome_new)
        error_type(change)
