"""
Programming Project 13
Ankit Khandelwal
15863
"""

from math import sqrt, asin
import numpy as np
from cmath import sin, cos, exp
from random import random
import scipy.sparse as sp

N = 7
psi = np.ndarray((2**N,1), complex) # N-qubit register

error_prob_percent = 10 
error = asin(sqrt(error_prob_percent/100))

H = 1/sqrt(2)*np.array([[1, 1],[1, -1]]) # Hadamard gate
I = np.eye(2)
EX = np.array(([cos(error), -1j*sin(error)], [-1j*sin(error), cos(error)]), complex)
EZ = np.array(([exp(-1j*error), 0], [0, exp(1j*error)]), complex)
X = np.array(([0, 1], [1, 0]), int)
Z = np.array(([1, 0], [0, -1]), int)
Y = 1j*X@Z

def kd(i,j):
    if i == j:
        return 1
    else:
        return 0

zero_ket = np.ndarray((2,1), int)
zero_ket[0][0] = 1
zero_ket[1][0] = 0
zero_bra = zero_ket.T

one_ket = np.ndarray((2,1), int)
one_ket[0][0] = 0
one_ket[1][0] = 1
one_bra = one_ket.T

zero_matrix = zero_ket@zero_bra
one_matrix = one_ket@one_bra

def C_NOTn(a,b):
    #part 1
    left = a-1
    right = N-a
    if right > 0:
        RIGHT = sp.eye(2**right, dtype=int)
    if left > 0:
        LEFT = sp.eye(2**left, dtype=int)
    if left == 0:
        LEFT = 1
    if right == 0:
        RIGHT = 1
    part1 = sp.kron(LEFT, sp.kron(zero_matrix, RIGHT))
    #part2
    if a < b:
        a = a
        b = b
        count = 1
    else:
        temp = a
        a = b
        b = temp
        count = 2
    left = a-1
    right = N-b
    center = N-left-right-2
    if right > 0:
        RIGHT = sp.eye(2**right, dtype=int)
    if left > 0:
        LEFT = sp.eye(2**left, dtype=int)
    if left == 0:
        LEFT = 1
    if right == 0:
        RIGHT = 1
    if center > 0:
        CENTER = sp.eye(2**center, dtype=int)
    if center == 0:
        CENTER = 1
    if count == 1:
        part2 = sp.kron(LEFT, sp.kron(one_matrix, sp.kron(CENTER, sp.kron(X, RIGHT))))
    if count == 2:
        part2 = sp.kron(LEFT, sp.kron(X, sp.kron(CENTER, sp.kron(one_matrix, RIGHT))))
    return part1 + part2

def multi_gate(gate, n):
    right = N-n
    left = N-right-1
    if right > 0:
        RIGHT = sp.eye(2**right, dtype=int)
    if left > 0:
        LEFT = sp.eye(2**left, dtype=int)
    if right == 0:
        return sp.kron(LEFT, gate)
    if left == 0:
        return sp.kron(gate, RIGHT)
    return sp.csr_matrix((sp.kron(LEFT, sp.kron(gate, RIGHT))))

repeat = 100
result = np.ndarray((1, repeat), np.ndarray)

for k in range(repeat):    
    comp_state = 0
    for i in range(2**N):
        if i == comp_state:
            psi[i] = 1
        else:
            psi[i] = 0
            
    '''      
    Replace I to add errors & can also change qubit number to which error is added.
    Will correct 1 bit flip error
    Will correct 1 phase flip error
    This mean can correct X, Y, Z on any 1 qubit
    Fail to correct 2 qubit errors depending on position of errors
    '''
    
    #for project 13 circuit (b)
    part1 = C_NOTn(7,5)@C_NOTn(2,6)@C_NOTn(3,1)@C_NOTn(4,7)@C_NOTn(2,5)@C_NOTn(3,7)@C_NOTn(2,1)@C_NOTn(4,6)@multi_gate(H,2)@multi_gate(H,3)@multi_gate(H,4)
    part2 = multi_gate(H,1)@multi_gate(H,2)@multi_gate(H,3)@multi_gate(H,4)@multi_gate(H,5)@multi_gate(H,6)@multi_gate(H,7)
    part3 = multi_gate(Z,1)@multi_gate(Z,2)@multi_gate(Z,3)@multi_gate(Z,4)@multi_gate(Z,5)@multi_gate(Z,6)@multi_gate(Z,7)
    part4 = multi_gate(H,1)@multi_gate(H,2)@multi_gate(H,3)@multi_gate(H,4)@multi_gate(H,5)@multi_gate(H,6)@multi_gate(H,7)
    
    psi = (part4@multi_gate(I, 2)@part3@multi_gate(I, 2)@part2@multi_gate(I, 2)@part1@multi_gate(I, 2)@psi)

    r = random()
    q = 0
    for i in range(2**N):
        q = q + abs(psi[i])**2
        if r < q:
            result[0][k] = (i)  
            break

final_result = np.zeros(2, dtype = int)
result_freq = np.zeros((2**N), int)
for i in range(2**N):
    result_freq[i] = np.count_nonzero(result[0][:] == i)
    b=[0]
    string = format(i, '0{}b'.format(N))
    if result_freq[i] > 0:
        for ii in range(N):
            b.append(int(string[ii]))
        if (b[4]+b[5]+b[6]+b[7])%2 == 0:
            S4 = 0
        if (b[4]+b[5]+b[6]+b[7])%2 == 1:
            S4 = 1
        if (b[2]+b[3]+b[6]+b[7])%2 == 0:
            S5 = 0
        if (b[2]+b[3]+b[6]+b[7])%2 == 1:
            S5 = 1
        if (b[1]+b[3]+b[5]+b[7])%2 == 0:
            S6 = 0
        if (b[1]+b[3]+b[5]+b[7])%2 == 1:
            S6 = 1
        if (b[4]+b[5]+b[6]+b[7]+b[1]+b[2]+b[3])%2 == 0:
            ZL = 0
        if (b[4]+b[5]+b[6]+b[7]+b[1]+b[2]+b[3])%2 == 1:
            ZL = 1
        
        if S4+S5+S6 > 0:
            ans = int(not(ZL))
        else:
            ans = (ZL)
        if ans == 0:
            final_result[0] = final_result[0] + result_freq[i]
        else:
            final_result[1] = final_result[1] + result_freq[i]
print('0 measured {} times and 1 measured {} times.'.format(final_result[0], final_result[1]))