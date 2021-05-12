"""
Programming Project 16
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
xnos = 3
znos = 4
dnos = 8
data_qubits = np.ndarray((dnos,2), list)
data_qubits[0][0] = 1
data_qubits[1][0] = 3
data_qubits[2][0] = 5
data_qubits[3][0] = 7
data_qubits[4][0] = 9
data_qubits[5][0] = 11
data_qubits[6][0] = 13
data_qubits[7][0] = 15
data_qubits[0][1] = [1,3] 
data_qubits[1][1] = [3,3]
data_qubits[2][1] = [5,3]
data_qubits[3][1] = [2,2]
data_qubits[4][1] = [4,2]
data_qubits[5][1] = [1,1]
data_qubits[6][1] = [3,1]
data_qubits[7][1] = [5,1]

X_stabilizers = np.ndarray((xnos,6), list)
X_stabilizers[0][0] = 6
X_stabilizers[0][1] = 1
X_stabilizers[0][2] = 'a'
X_stabilizers[0][3] = 7
X_stabilizers[0][4] = 11
X_stabilizers[0][5] = [1,2]
X_stabilizers[1][0] = 8
X_stabilizers[1][1] = 3
X_stabilizers[1][2] = 7
X_stabilizers[1][3] = 9
X_stabilizers[1][4] = 13
X_stabilizers[1][5] = [3,2]
X_stabilizers[2][0] = 10
X_stabilizers[2][1] = 5
X_stabilizers[2][2] = 9
X_stabilizers[2][3] = 'a'
X_stabilizers[2][4] = 15
X_stabilizers[2][5] = [5,2]

Z_stabilizers = np.ndarray((znos,6), list)
Z_stabilizers[0][0] = 2
Z_stabilizers[0][1] = 'a'
Z_stabilizers[0][2] = 1
Z_stabilizers[0][3] = 3
Z_stabilizers[0][4] = 7
Z_stabilizers[0][5] = [2,3]
Z_stabilizers[1][0] = 4
Z_stabilizers[1][1] = 'a'
Z_stabilizers[1][2] = 3
Z_stabilizers[1][3] = 5
Z_stabilizers[1][4] = 9
Z_stabilizers[1][5] = [4,3]
Z_stabilizers[2][0] = 12
Z_stabilizers[2][1] = 7
Z_stabilizers[2][2] = 11
Z_stabilizers[2][3] = 13
Z_stabilizers[2][4] = 'a'
Z_stabilizers[2][5] = [2,1]
Z_stabilizers[3][0] = 14
Z_stabilizers[3][1] = 9
Z_stabilizers[3][2] = 13
Z_stabilizers[3][3] = 15
Z_stabilizers[3][4] = 'a'
Z_stabilizers[3][5] = [4,1]

ZL_cycle = [3, 13]
XL_cycle = [1, 7, 13, 15]
line1x=[]
line1y=[]
line2x=[]
line2y=[]
for i in range(len(XL_cycle)):
    line1x.append(data_qubits[np.argwhere(data_qubits[:,0] == XL_cycle[i])[0][0]][1][0])
    line1y.append(data_qubits[np.argwhere(data_qubits[:,0] == XL_cycle[i])[0][0]][1][1])
for i in range(len(ZL_cycle)):
    line2x.append(data_qubits[np.argwhere(data_qubits[:,0] == ZL_cycle[i])[0][0]][1][0])
    line2y.append(data_qubits[np.argwhere(data_qubits[:,0] == ZL_cycle[i])[0][0]][1][1])
plt.plot(line1x, line1y, 'k--', label = 'XL Cycle')
plt.plot(line2x, line2y, 'k-', label = 'ZL Cycle')
fl = plt.legend(loc=(1.04,.8), fontsize = 15)
plt.gca().add_artist(fl)
for n in range(dnos):
    d = plt.scatter(data_qubits[n][1][0], data_qubits[n][1][1], c='r', s=1000 )
for n in range(xnos):
    x = plt.scatter(X_stabilizers[n][5][0], X_stabilizers[n][5][1], c='b', s=1000 )
for n in range(znos):
    z = plt.scatter(Z_stabilizers[n][5][0], Z_stabilizers[n][5][1], c='g', s=1000 )
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
                             Z_stabilizers[:,0], [N+1-N])).reshape(1,xnos + znos +1)
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



repeat = 100
results = 5*np.ones((N), int)

def stab_cycle(psi, error):
    if error == True:
        error_prob = 0.02
        error_prob_per_gate = error_prob/3
        gates = ['I', 'X', 'Y', 'Z']
        for i in range(dnos):
            a = np.random.choice(gates, 1, p=[1-error_prob,error_prob_per_gate,error_prob_per_gate,error_prob_per_gate])
            b = a[0]
            if b == 'I': 
                continue
            elif b == 'X':
                psi = multi_gate(X, data_qubits[i][0])@psi
                #print(b,data_qubits[i][0])
            elif b == 'Y':
                psi = multi_gate(Y, data_qubits[i][0])@psi
                #print(b,data_qubits[i][0])
            elif b == 'Z':
                psi = multi_gate(Z, data_qubits[i][0])@psi
                #print(b,data_qubits[i][0])
    #print('*')
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
    zl = Z_stabilizers[-1,0]
    measure_qubits = [N+1-zl]
    for n in measure_qubits:
        prob[n][0] = (((PP_0[n]@psi).conj().T)@(PP_0[n]@psi)).real
        prob[n][1] = (((PP_1[n]@psi).conj().T)@(PP_1[n]@psi)).real
        r = choices((0,1), (prob[n][0], prob[n][1]))
        r = r[0]
        if r == 0:
            psi = (PP_0[n]@psi)/sqrt(prob[n][0])
        else:
            psi = (PP_1[n]@psi)/sqrt(prob[n][1])
    r1 = r
    for i in range(len(ZL_cycle)):
        psi = C_NOTn(ZL_cycle[i], zl)@psi
    for n in measure_qubits:
        prob[n][0] = (((PP_0[n]@psi).conj().T)@(PP_0[n]@psi)).real
        prob[n][1] = (((PP_1[n]@psi).conj().T)@(PP_1[n]@psi)).real
        r = choices((0,1), (prob[n][0], prob[n][1]))
        r = r[0]
        if r == 0:
            psi = (PP_0[n]@psi)/sqrt(prob[n][0])
        else:
            psi = (PP_1[n]@psi)/sqrt(prob[n][1])
    r2 = r
    if r2 == r1: ZL = 0
    else: ZL = 1
    return ZL, psi

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

error_table = np.zeros((4**dnos, 1+xnos+znos+1), int)
def numToBase(n, b):
    digits = np.zeros(dnos, int)
    if n == 0:
        return digits
    i = 0
    while n:
        digits[i] = (int(n%b))
        n //= b
        i = i+1
    return digits[::-1]
for i in range(4**dnos):
    temp = numToBase(i, 4)
    error_table[i][-2] = i 
    error_table[i][-1] = np.count_nonzero(temp == 0)
    for k in range(dnos):
        if temp[k] == 1 or temp[k] == 3:
            qubit_loc = np.argwhere((Z_stabilizers[:,1:-1]) == (data_qubits[k][0]))
            for l in range(len(qubit_loc)):
               error_table[i][qubit_loc[l][0]+xnos] = 1 - error_table[i][qubit_loc[l][0]+xnos]
        if temp[k] == 2 or temp[k] == 3:
            qubit_loc = np.argwhere((X_stabilizers[:,1:-1]) == (data_qubits[k][0]))
            for l in range(len(qubit_loc)):
               error_table[i][qubit_loc[l][0]] = 1 - error_table[i][qubit_loc[l][0]]
error_table = error_table[error_table[:,-1].argsort()[::-1]]


def error_type(change):
    index = error_table[:,0:xnos+znos].tolist().index(change.tolist())
    gate_list = numToBase(error_table[index][xnos+znos], 4)
    for i in range(len(gate_list)):
        if gate_list[i] == 0:
            continue
        elif gate_list[i] == 1:
            #print('X error on {} qubit.'.format(data_qubits[i][0]))
            error_count[[np.argwhere(error_count[:,0] == data_qubits[i][0])][0][0][0]][1] = \
                1 - error_count[[np.argwhere(error_count[:,0] == data_qubits[i][0])][0][0][0]][1]
        elif gate_list[i] == 2:
            #print('Z error on {} qubit.'.format(data_qubits[i][0]))
            error_count[[np.argwhere(error_count[:,0] == data_qubits[i][0])][0][0][0]][2] = \
                1 - error_count[[np.argwhere(error_count[:,0] == data_qubits[i][0])][0][0][0]][2]
        elif gate_list[i] == 3:
            #print('Y error on {} qubit.'.format(data_qubits[i][0]))
            error_count[[np.argwhere(error_count[:,0] == data_qubits[i][0])][0][0][0]][1] = \
                1 - error_count[[np.argwhere(error_count[:,0] == data_qubits[i][0])][0][0][0]][1]
            error_count[[np.argwhere(error_count[:,0] == data_qubits[i][0])][0][0][0]][2] = \
                1 - error_count[[np.argwhere(error_count[:,0] == data_qubits[i][0])][0][0][0]][2]
    E_Z = []
    E_X = []
    for i in ZL_cycle:
        E_Z.append(error_count[[np.argwhere(error_count[:,0] == i)][0][0][0]][1])
    for i in XL_cycle:
        E_X.append(error_count[[np.argwhere(error_count[:,0] == i)][0][0][0]][2])
    E_Z = sum(E_Z)
    E_X = sum(E_X)
    return E_Z, E_X

ans = np.zeros(repeat, int)
for ii in range(repeat):
    computation = ['H', 'T', 'H'] 
    '''
    It can be ['H', 'H'] or ['H', 'Z', 'H'] or ['H', 'T', 'H']
    '''
    if ii == 0: print('Computation = ', computation)
    error_count = np.zeros((dnos,3), int)
    for i in range(dnos):
        error_count[i][0] = data_qubits[i][0]
    final_computation = []
    # psi is computational basis state
    comp_state = randint(1, 2**N)
    for i in range(2**N):
        if i == comp_state:
            psi[i] = 1
        else:
            psi[i] = 0
    pending_gate = 0
    
    syndrome_old, psi = stab_cycle(psi, error=0) # error must be zero
    #print(syndrome_old)
    Z_l_init, psi = meas_ZL(psi)
    if Z_l_init == 1:
        pending_gate = 1
    syndrome_new, psi = stab_cycle(psi, 1) # 1 for errors
    #print(syndrome_new,'++')
    change = abs(syndrome_new - syndrome_old)
    syndrome_old = np.copy(syndrome_new)
    E_Z, E_X = error_type(change)

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
            if (E_Z + E_X) % 2 != 0:
                if (gate != HL_mat).nnz == 0 :
                    gate = HL_mat_er
            if E_Z % 2 != 0:
                if (gate != TL_mat).nnz == 0 :
                    gate = TL_daggar_mat
                elif (gate != TL_daggar_mat).nnz == 0 :
                    gate = TL_mat
            psi = gate@psi
            syndrome_new, psi = stab_cycle(psi, 1) # 1 for errors
            #print(syndrome_new,'*')
            change = abs(syndrome_new - syndrome_old)
            syndrome_old = np.copy(syndrome_new)
            E_Z, E_X = error_type(change)

    Z_l_final, psi = meas_ZL(psi)
    
    correct_error = []
    for i in ZL_cycle:
        correct_error.append(error_count[[np.argwhere(error_count[:,0] == i)][0][0][0]][1])
    
    if sum(correct_error)%2 != 0 :
        Z_l_final = 1 - Z_l_final
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
