"""
Programming Project 12
Ankit Khandelwal
15863
"""

import cmath
from cmath import sin, cos, exp
from math import sqrt, pi, asin
from random import random

import numpy as np

N = 3
psi = np.ndarray((2 ** N, 1), complex)  # N-qubit register
comp_state = 0
for i in range(2 ** N):
    if i == comp_state:
        psi[i] = 1
    else:
        psi[i] = 0
psi_state_0 = np.copy(psi)

j = cmath.sqrt(-1)
error_prob_percent = 10
error = asin(sqrt(error_prob_percent / 100))

H = 1 / sqrt(2) * np.array([[1, 1], [1, -1]])  # Hadamard gate
I = np.eye(2, dtype=int)
EX = np.array(([cos(error), -j * sin(error)], [-j * sin(error), cos(error)]), complex)
EZ = np.array(([exp(-j * error), 0], [0, exp(j * error)]), complex)
X = np.array(([0, 1], [1, 0]), int)
Z = np.array(([1, 0], [0, -1]), int)


def kd(i, j):
    if i == j:
        return 1
    else:
        return 0


C_NOT = np.zeros((4, 4), int)
C_NOT[0][0] = 1
C_NOT[1][1] = 1
C_NOT[2][3] = 1
C_NOT[3][2] = 1


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


def multi_gate(gate, n):
    right = N - n
    left = N - right - 1
    if right > 0:
        RIGHT = np.eye(2 ** right, dtype=int)
    if left > 0:
        LEFT = np.eye(2 ** left, dtype=int)
    if right == 0:
        return np.kron(LEFT, gate)
    if left == 0:
        return np.kron(gate, RIGHT)
    return (np.kron(LEFT, np.kron(gate, RIGHT)))


def R(th):  # Phase shift gate
    return np.array([[1, 0], [0, cmath.exp(th * j)]], complex)


T = R(pi / 4)
S = R(pi / 2)
T_dag = R(-pi / 4)


def Toffoli(a, b, c):
    return C_NOTn(a, b) @ multi_gate(H, c) @ multi_gate(S, a) @ C_NOTn(b, c) @ C_NOTn(c, a) @ multi_gate(T_dag,
                                                                                                         a) @ multi_gate(
        T_dag, b) @ multi_gate(T, c) @ C_NOTn(b, a) @ multi_gate(T_dag, a) @ C_NOTn(b, c) @ C_NOTn(c, a) @ C_NOTn(a,
                                                                                                                  b) @ multi_gate(
        T_dag, a) @ multi_gate(T, b) @ multi_gate(T, c) @ multi_gate(H, c)


repeat = 100
result = np.ndarray((2, repeat), np.ndarray)

ques = 2

for k in range(repeat):
    PSI = []
    psi = np.copy(psi_state_0)
    '''
     Replace I to add errors & can also change qubit number to which error is added.
     Will correct 1 bit flip error
     Will not correct phase flip error
     Fail to correct 2 bit flip errors depending on position of errors
    '''
    # for project 12 circuit (b)
    PSI.append(multi_gate(I, 3) @ psi)
    psi = np.copy(psi_state_0)

    # for project 12 circuit (d)
    H_corrected = C_NOTn(1, 2) @ C_NOTn(1, 3) @ Toffoli(3, 2, 1) @ multi_gate(H, 1) @ Toffoli(3, 2, 1) @ C_NOTn(1,
                                                                                                                3) @ C_NOTn(
        1, 2)
    PSI.append(multi_gate(I, 2) @ H_corrected @ multi_gate(I, 2) @ H_corrected @ multi_gate(I, 3) @ psi)

    for n in range(ques):
        r = random()
        q = 0
        for i in range(2 ** N):
            q = q + abs(PSI[n][i]) ** 2
            if r < q:
                result[n][k] = (i)
                break

for n in range(ques):
    result_freq = np.zeros((2 ** N), int)
    if n == 0:
        print('\nfor project 12 circuit (b)')
    else:
        print('\nfor project 12 circuit (d)')
    for i in range(2 ** N):
        result_freq[i] = np.count_nonzero(result[n][:] == i)
        print(format(i, '0{}b'.format(N)), 'occured', result_freq[i], 'times.', (result_freq[i] / repeat * 100),
              'percent.')
    print('***')

    corrected_result_freq = np.zeros((2), int)
    for i in range(2 ** N):
        jj = str(format(i, '0{}b'.format(N)))
        sumx = 0
        for jjj in range(3):
            sumx = sumx + int(jj[jjj])
        if sumx < 2:
            corrected_result_freq[0] = corrected_result_freq[0] + result_freq[i]
        else:
            corrected_result_freq[1] = corrected_result_freq[1] + result_freq[i]

    print('0 occured', corrected_result_freq[0], 'times.')
    print('1 occured', corrected_result_freq[1], 'times.')
    print('===========')
