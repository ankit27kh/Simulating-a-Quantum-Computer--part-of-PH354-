"""
Programming Project 17
Ankit Khandelwal
15863
"""

from math import sqrt, pi
import numpy as np
from cmath import exp
from random import choices, randint
import scipy.sparse as sp
import matplotlib.pyplot as plt
from time import perf_counter

t1= perf_counter()

N = 15
xnos = 6
znos = 2
dnos = 7
data_qubits = np.ndarray((dnos,2), list)
data_qubits[0][0] = 2
data_qubits[1][0] = 4
data_qubits[2][0] = 6
data_qubits[3][0] = 8
data_qubits[4][0] = 10
data_qubits[5][0] = 12
data_qubits[6][0] = 14
data_qubits[0][1] = [2,3] 
data_qubits[1][1] = [4,3]
data_qubits[2][1] = [1,2]
data_qubits[3][1] = [3,2]
data_qubits[4][1] = [5,2]
data_qubits[5][1] = [2,1]
data_qubits[6][1] = [4,1]

Z_stabilizers = np.ndarray((znos,6), list)
Z_stabilizers[0][0] = 7
Z_stabilizers[0][1] = 2
Z_stabilizers[0][2] = 6
Z_stabilizers[0][3] = 8
Z_stabilizers[0][4] = 12
Z_stabilizers[0][5] = [2,2]
Z_stabilizers[1][0] = 9
Z_stabilizers[1][1] = 4
Z_stabilizers[1][2] = 8
Z_stabilizers[1][3] = 10
Z_stabilizers[1][4] = 14
Z_stabilizers[1][5] = [4,2]

X_stabilizers = np.ndarray((xnos,6), list)
X_stabilizers[0][0] = 1
X_stabilizers[0][1] = 'a'
X_stabilizers[0][2] = 'a'
X_stabilizers[0][3] = 2
X_stabilizers[0][4] = 6
X_stabilizers[0][5] = [1,3]
X_stabilizers[1][0] = 3
X_stabilizers[1][1] = 'a'
X_stabilizers[1][2] = 2
X_stabilizers[1][3] = 4
X_stabilizers[1][4] = 8
X_stabilizers[1][5] = [3,3]
X_stabilizers[2][0] = 5
X_stabilizers[2][1] = 'a'
X_stabilizers[2][2] = 4
X_stabilizers[2][3] = 'a'
X_stabilizers[2][4] = 10
X_stabilizers[2][5] = [5,3]
X_stabilizers[3][0] = 11
X_stabilizers[3][1] = 6
X_stabilizers[3][2] = 'a'
X_stabilizers[3][3] = 12
X_stabilizers[3][4] = 'a'
X_stabilizers[3][5] = [1,1]
X_stabilizers[4][0] = 13
X_stabilizers[4][1] = 8
X_stabilizers[4][2] = 12
X_stabilizers[4][3] = 14
X_stabilizers[4][4] = 'a'
X_stabilizers[4][5] = [3,1]
X_stabilizers[5][0] = 15
X_stabilizers[5][1] = 10
X_stabilizers[5][2] = 14
X_stabilizers[5][3] = 'a'
X_stabilizers[5][4] = 'a'
X_stabilizers[5][5] = [5,1]

defect_z_stabilizer = 7
Z_stabilizers_without_defect = np.delete(Z_stabilizers, np.argwhere(Z_stabilizers[0,:] == defect_z_stabilizer)[0][0], 0)
Z_stabilizers_defect = (Z_stabilizers[np.argwhere(Z_stabilizers[0,:] == defect_z_stabilizer)[0][0], :]).reshape(1,6)

ZL_cycle = [2,8,12,6]
XL_cycle = [8,10]
line1x=[]
line1y=[]
line2x=[data_qubits[np.argwhere(data_qubits[:,0] == ZL_cycle[-1])[0][0]][1][0]]
line2y=[data_qubits[np.argwhere(data_qubits[:,0] == ZL_cycle[-1])[0][0]][1][1]]
for i in range(len(XL_cycle)):
    line1x.append(data_qubits[np.argwhere(data_qubits[:,0] == XL_cycle[i])[0][0]][1][0])
    line1y.append(data_qubits[np.argwhere(data_qubits[:,0] == XL_cycle[i])[0][0]][1][1])
for i in range(len(ZL_cycle)):
    line2x.append(data_qubits[np.argwhere(data_qubits[:,0] == ZL_cycle[i])[0][0]][1][0])
    line2y.append(data_qubits[np.argwhere(data_qubits[:,0] == ZL_cycle[i])[0][0]][1][1])
plt.plot(line1x, line1y, 'k--', label = 'XL Cycle')
plt.plot(line2x, line2y, 'k-', label = 'ZL Cycle')
for n in range(dnos):
    d = plt.scatter(data_qubits[n][1][0], data_qubits[n][1][1], c='r', s=1000 )
for n in range(xnos):
    x = plt.scatter(X_stabilizers[n][5][0], X_stabilizers[n][5][1], c='b', s=1000 )
for n in range(znos):
    z = plt.scatter(Z_stabilizers[n][5][0], Z_stabilizers[n][5][1], c='g', s=1000 )
plt.plot(Z_stabilizers_defect[0][5][0], Z_stabilizers_defect[0][5][1], 'ko', label='Defect')
fl = plt.legend(loc=(1.04,.8), fontsize = 15)
plt.gca().add_artist(fl)
plt.title('Surface Code Computer')
plt.xticks([1,2,3,4,5])
plt.yticks([1,2,3])
plt.legend((d, x, z), ('Data Qubits', 'X-Stabilizers', 'Z-Stabilizers'), fontsize = 12, scatterpoints=1, loc=(1.04,0), markerscale=.4)
plt.show()

psi = np.ndarray((2**N,1), complex) # N-qubit register

H = 1/sqrt(2)*np.array([[1, 1],[1, -1]]) # Hadamard gate
X = np.array(([0, 1], [1, 0]), int)
Z = np.array(([1, 0], [0, -1]), int)
Y = X@Z*1j

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

def P_0(n):
    P0 = np.ndarray((1, 2**N), int)
    for k in range(2**N):
            jj = str(format(k, '0{}b'.format(N)))
            if jj[N-n] == '0':
                P0[0][k] = 1
            else:
                P0[0][k] = 0
    return P0
measure_qubits_a = np.concatenate(((N+1) - X_stabilizers[:,0], (N+1) -\
                             Z_stabilizers[:,0])).reshape(1,xnos + znos)
                            # Needs to be custom in each code in (N+1-i) format
                            # where i is the qubit to be measured
PP_0 = np.ndarray((N+1), np.ndarray)
for nn in measure_qubits_a:
    for n in nn:    
        PP_0[n] = (P_0(n))

def P_1(n):
    P1 = np.ndarray((1, 2**N), int)
    for k in range(2**N):           
            P1[0][k] = kd(k, k) - PP_0[n][0][k]  
    return sp.csr_matrix(sp.diags(P1[0,:], 0))

PP_1 = np.ndarray((N+1), np.ndarray)
for nn in measure_qubits_a:
    for n in nn:    
        PP_1[n] = (P_1(n))
        PP_0[n] = sp.csr_matrix(sp.diags(PP_0[n][0,:],0))

prob = np.zeros((N+1,2), float)

measure_qubits = np.concatenate(((N+1) - X_stabilizers[:,0], (N+1) -\
                             Z_stabilizers[:,0])).reshape(1,xnos + znos)



repeat = 10
results = 5*np.ones((N), int)

def stab_cycle(psi, defect, defect_z_stabilizer, znos, Z_stabilizers):
    if defect == True:
        Z_stabilizers = Z_stabilizers_without_defect
        znos = znos - 1
    else:
        Z_stabilizers = Z_stabilizers
        znos = znos
    syndrome = np.zeros(xnos+znos, int)
    # 1 Initialize    
    for nn in measure_qubits:
        for n in nn:
            prob[n][0] = (((PP_0[n]@psi).conj().T)@(PP_0[n]@psi)).real
            prob[n][1] = (((PP_1[n]@psi).conj().T)@(PP_1[n]@psi)).real
            r = choices((0,1), (prob[n][0], prob[n][1]))
            r = r[0]
            if r == 0:
                psi = (PP_0[n]@psi)/sqrt(prob[n][0])
            else:
                psi = (PP_1[n]@psi)/sqrt(prob[n][1])
            results[N-n] = r
    syndome_qubits = np.concatenate((X_stabilizers[:,0],\
                           Z_stabilizers[:,0])).reshape(1,xnos + znos)
    for nn in syndome_qubits:
        for n in nn:
            if results[n-1] == 1:
                psi = multi_gate(X, n)@psi
            
    # 2 H
    for i in range(xnos):
        psi = multi_gate(H, X_stabilizers[i][0])@psi
    
    # 3 Up
    for i in range(xnos):
        if X_stabilizers[i][1] == 'a':
            continue
        psi = C_NOTn(X_stabilizers[i][0], X_stabilizers[i][1])@psi
    
    for i in range(znos):
        if Z_stabilizers[i][1] == 'a':
            continue
        psi = C_NOTn(Z_stabilizers[i][1], Z_stabilizers[i][0])@psi

    # 4 Left
    for i in range(xnos):
        if X_stabilizers[i][2] == 'a':
            continue
        psi = C_NOTn(X_stabilizers[i][0], X_stabilizers[i][2])@psi
    
    for i in range(znos):
        if Z_stabilizers[i][2] == 'a':
            continue
        psi = C_NOTn(Z_stabilizers[i][2], Z_stabilizers[i][0])@psi
        
    # 5 Right
    for i in range(xnos):
        if X_stabilizers[i][3] == 'a':
            continue        
        psi = C_NOTn(X_stabilizers[i][0], X_stabilizers[i][3])@psi
    
    for i in range(znos):
        if Z_stabilizers[i][3] == 'a':
            continue
        psi = C_NOTn(Z_stabilizers[i][3], Z_stabilizers[i][0])@psi

    # 6 Down
    for i in range(xnos):
        if X_stabilizers[i][4] == 'a':
            continue        
        psi = C_NOTn(X_stabilizers[i][0], X_stabilizers[i][4])@psi
    
    for i in range(znos):
        if Z_stabilizers[i][4] == 'a':
            continue        
        psi = C_NOTn(Z_stabilizers[i][4], Z_stabilizers[i][0])@psi
        
    # 7 H
    for i in range(xnos):
        psi = multi_gate(H, X_stabilizers[i][0])@psi
        
    # 8 Measure
    for nn in measure_qubits:
        for n in nn:
            prob[n][0] = (((PP_0[n]@psi).conj().T)@(PP_0[n]@psi)).real
            prob[n][1] = (((PP_1[n]@psi).conj().T)@(PP_1[n]@psi)).real
            r = choices((0,1), (prob[n][0], prob[n][1]))
            r = r[0]
            if r == 0:
                psi = (PP_0[n]@psi)/sqrt(prob[n][0])
            else:
                psi = (PP_1[n]@psi)/sqrt(prob[n][1])
            results[N-n] = r
    
    for nn in syndome_qubits:
        i = 0
        for n in nn:
            syndrome[i] = (results[n-1])
            i = i+1
    return syndrome, psi

def meas_ZL(psi):
    syndrome, psi = stab_cycle(psi, 0, defect_z_stabilizer, znos, Z_stabilizers)
    n = np.argwhere(Z_stabilizers[:,0] == defect_z_stabilizer)[0][0]
    r = syndrome[xnos+n]
    return r, psi

XL_mat = sp.eye(2**N)
for i in range(len(XL_cycle)):
    XL_mat = sp.csr_matrix(multi_gate(X, XL_cycle[i])@XL_mat)
ZL_mat = sp.eye(2**N)
for i in range(len(ZL_cycle)):
    ZL_mat = sp.csr_matrix(multi_gate(Z, ZL_cycle[i])@ZL_mat)

HL_mat = (XL_mat + ZL_mat)/sqrt(2)
A_plus = (1+exp(pi*1j/4))/2
A_minus = (1-exp(pi*1j/4))/2
IL_mat = sp.eye(2**N)
TL_mat = A_plus*IL_mat + A_minus*ZL_mat
TL_daggar_mat = A_plus.conjugate()*IL_mat + A_minus.conjugate()*ZL_mat
YL_mat = 1j*XL_mat@ZL_mat
HL_mat_er = (XL_mat - ZL_mat)/sqrt(2)
available_gates = ['X', 'Y', 'Z', 'H', 'T', 'Tdag']

ans = np.zeros(repeat, int)
for ii in range(repeat):
    computation = ['H', 'H', 'H', 'H', 'I', 'H', 'H', 'H', 'H']
    if ii == 0: print('Computation = ', computation)
    final_computation = []
    # psi is computational basis state
    comp_state = randint(1, 2**N)
    for i in range(2**N):
        if i == comp_state:
            psi[i] = 1
        else:
            psi[i] = 0
    pending_gate = 0
    
    Z_l_init, psi = meas_ZL(psi)
    if Z_l_init == 1:
        pending_gate = 1
        
    syndrome_new, psi = stab_cycle(psi, 1, defect_z_stabilizer, znos, Z_stabilizers)

    for gate in computation:
        if pending_gate == 0:
            if gate == 'X':
                pending_gate = 1
            elif gate == 'Y':
                pending_gate = 2
            elif gate == 'Z':
                pending_gate = 3
            elif gate == 'H':
                pending_gate = 0
                final_computation.append(HL_mat)
            elif gate == 'T':
                pending_gate = 0
                final_computation.append(TL_mat)
            elif gate == 'Tdag':
                pending_gate = 0
                final_computation.append(TL_daggar_mat)
        elif pending_gate == 1:
            if gate == 'X':
                pending_gate = 0
            elif gate == 'Y':
                pending_gate = 2
                final_computation.append(IL_mat*1j)
            elif gate == 'Z':
                pending_gate = 3
                final_computation.append(YL_mat*1j)
            elif gate == 'H':
                pending_gate = 2
                final_computation.append(HL_mat)
            elif gate == 'T':
                pending_gate = 1
                final_computation.append(TL_daggar_mat)
            elif gate == 'Tdag':
                pending_gate = 1
                final_computation.append(TL_mat)
        elif pending_gate == 2:
            if gate == 'X':
                pending_gate = 3
                final_computation.apend(IL_mat*1j)
            elif gate == 'Y':
                pending_gate = 1
                final_computation.append(-IL_mat*1j)
            elif gate == 'Z':
                pending_gate = 0
            elif gate == 'H':
                pending_gate = 1
                final_computation.append(HL_mat)
            elif gate == 'T':
                pending_gate = 2
                final_computation.append(TL_mat)
            elif gate == 'Tdag':
                pending_gate = 2
                final_computation.append(TL_daggar_mat)
        elif pending_gate == 3:
            if gate == 'X':
                pending_gate = 2
                final_computation.append(-IL_mat*1j)
            elif gate == 'Y':
                pending_gate = 0
            elif gate == 'Z':
                pending_gate = 1
                final_computation.append(IL_mat*1j)
            elif gate == 'H':
                pending_gate = 3
                final_computation.append(-HL_mat)
            elif gate == 'T':
                pending_gate = 3
                final_computation.append(TL_daggar_mat)
            elif gate == 'Tdag':
                pending_gate = 3
                final_computation.append(TL_mat)
    
    for gate in final_computation:
            psi = gate@psi
            syndrome_new, psi = stab_cycle(psi, 1, defect_z_stabilizer, znos, Z_stabilizers)
     #       print(syndrome_new)
    
    Z_l_final, psi = meas_ZL(psi)
    
    if pending_gate == 1 or pending_gate == 3:
        if Z_l_final == 1:
            Z_l_final = 0
        else:
            Z_l_final = 1

    ans[ii] = Z_l_final

print('0 percent = ', np.count_nonzero(ans == 0)/repeat*100)
print('1 percent = ', np.count_nonzero(ans == 1)/repeat*100)
t2 = perf_counter()
print('Time taken :(in seconds)', t2-t1, 'for {} computations.'.format(repeat))
