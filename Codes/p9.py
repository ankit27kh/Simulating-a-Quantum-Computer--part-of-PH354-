"""
Programming Project 9
Ankit Khandelwal
15863
"""

from math import sqrt, pi
import numpy as np
import cmath
from scipy import sparse as sp

N = 3

def R(th):  # Phase shift gate
    return np.array([[1, 0],[0, cmath.exp(th*1j)]], complex)

H = 1/sqrt(2)*np.array([[1, 1],[1, -1]]) # Hadamard gate

def multi_gate(gate, n): # Gate 'gate' on qubit 'n' in N qubit circuit
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
    return sp.csr_matrix(sp.kron(LEFT, sp.kron(gate, RIGHT)))

T = R(pi/4) 
S = R(pi/2)
T_dag = R(-pi/4)

def kd(i,j): # Kronecker delta
    if i == j:
        return 1
    else:
        return 0
 
C_NOT = np.zeros((4, 4), int)
C_NOT[0][0] = 1
C_NOT[1][1] = 1
C_NOT[2][3] = 1
C_NOT[3][2] = 1

def C_NOTn(a,b): # CNOT gate for N qubits with (control, target)
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
    return sp.csr_matrix(C_NOTn)

# Toffoli Gate Decomposition
Tof = np.eye((2**3), dtype=int)
Tof[6][6] = 0
Tof[7][7] = 0
Tof[6][7] = 1
Tof[7][6] = 1
Toffoli = (C_NOTn(1,2)@multi_gate(H,3)@multi_gate(S,1)@C_NOTn(2,3)@C_NOTn(3,1)@multi_gate(T_dag,1)@multi_gate(T_dag,2)@multi_gate(T,3)@C_NOTn(2,1)@multi_gate(T_dag,1)@C_NOTn(2,3)@C_NOTn(3,1)@C_NOTn(1,2)@multi_gate(T_dag,1)@multi_gate(T,2)@multi_gate(T,3)@multi_gate(H,3)).real
print('Difference b/w decomposed and original Toffoli gate:')
print(Toffoli-Tof)

#Fig Second circuit for project 9
N = 2

C_X = multi_gate(H,1)@multi_gate(H,2)@C_NOTn(2,1)@multi_gate(H,1)@multi_gate(H,2)
print('\nDifference b/w matrix from circuit in 11c and C-NOT:')
print(C_X.todense()-C_NOT)
