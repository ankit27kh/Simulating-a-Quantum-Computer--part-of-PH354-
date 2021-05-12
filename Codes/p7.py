"""
Programming Project 7
Ankit Khandelwal
15863
"""

from math import log10, gcd, sqrt, pi, floor
import numpy as np
import cmath
from scipy import sparse as sp
from random import random, randint

N = 7 # Number of qubits
l=3
r=4
C=15
stop = 0
if C%2 == 0:
    print('Factor = 2')
    stop = 1

for x in range(2, int(sqrt(C))+1):
    y = log10(C)/log10(x)
    if int(y) == y:
        #print(x, y, x**y, C)
        print('Factor = ', x)
        stop = 1
        break
    
a = randint(2, C-1)

if gcd(a, C) > 1:
    print('Factor = ', gcd(a, C))
    stop = 1

if stop == 0:
    def R(th):  # Phase shift gate
        return np.array([[1, 0],[0, cmath.exp(th*1j)]], complex)
        
    def kd(i, k): # kronecker delta
        if i == k:
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
    
    def C_Rn(a, b, th, N): #Control phase shift gate with a controlling b in N qubit system
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
            part2 = sp.kron(LEFT, sp.kron(one_matrix, sp.kron(CENTER, sp.kron(R(th), RIGHT))))
        if count == 2:
            part2 = sp.kron(LEFT, sp.kron(R(th), sp.kron(CENTER, sp.kron(one_matrix, RIGHT))))
        return part1 + part2
        
    A = []
    for i in range(l):
        A.append(a**(2**i)%C)
    
    def f(l, r, n): # Matrix to compute f-register
        f = np.zeros((2**N, 2**N), int)
        for k in range(2**N):
            ii = str(format(k, '0{}b'.format(N)))    
            if (ii[l-n-1]) ==  '0':
                f[k][k] = 1
            else:
                ii1 = ii[:l]
                ii2 = ii[l:]
                ff = int(ii2, 2)
                if ff >= C:
                    f[k][k] = 1
                else:
                    ff = (A[n]*ff) % C
                    ii2 = str(format(ff, '0{}b'.format(r)))
                    ii = ii1 + ii2
                    f[int(ii, 2)][k] = 1
        return sp.csr_matrix(f)
    
    H = 1/sqrt(2)*np.array([[1, 1],[1, -1]]) # Hadamard gate
    
    def HH(n): # Hadamard gate on qubit 'n' in N qubit circuit
        right = N-n
        left = N-right-1
        if right > 0:
            RIGHT = sp.eye(2**right, dtype=int)
        if left > 0:
            LEFT = sp.eye(2**left, dtype=int)
        if right == 0:
            return sp.kron(LEFT, H)
        if left == 0:
            return sp.kron(H, RIGHT)
        return sp.csr_matrix(sp.kron(LEFT, sp.kron(H, RIGHT)))
    
    fff=[] # f-register
    repeat = 10
    xxx=np.ndarray((1,repeat)) # x-register
    for ii in range(repeat):
        psi = np.ndarray((2**N,1), complex) # N-qubit register
        comp_state = 1
        for i in range(2**N):
            if i == comp_state:
                psi[i] = 1
            else:
                psi[i] = 0
        # Step 1
        psi = HH(3)@HH(2)@HH(1)@psi
        psi = f(l, r, 2)@f(l, r, 1)@f(l, r, 0)@psi

        # IQFT
        psi = HH(3)@C_Rn(2, 3, pi/2, N)@HH(2)@C_Rn(1, 3, pi/4, N)@C_Rn(1, 2, pi/2, N)@HH(1)@psi

        # Measurment
        rand = random()
        q = 0
        for i in range(2**N):
            q = q + abs(psi[i])**2
            if rand < q:
                result = i  
                break
        result = str(format(result, '0{}b'.format(N)))
        #print(result)
        
        fff.append(int(result[3:], 2))
        x_1 = result[:3]
        xxx[0][ii] = (int(x_1[::-1], 2)) # x2_bar x1_bar x0_bar
    
    xxx = xxx / 2**l # x_bar / 2^l
    
    def cf(x, k): # Continued fraction of x till k terms
        cf = []
        q = floor(x)
        cf.append(q)
        x = x-q
        i = 0
        while x!=0 and i<k:
            q = floor(1/x)
            cf.append(q)
            x = 1/x - q
            i = i+1
        return cf 
    
    kk = 10 # Max no. of terms in continued fraction
    sol = 0
    for ii in range(repeat):
        qq =  (cf(xxx[0][ii], kk))
        #print(qq)
        if len(qq) == 1:
            continue
        h = [qq[0], qq[0]*qq[1]+1]
        k = [1, qq[1]]
        for i in range(2, len(qq)):
            h.append(qq[i]*h[-1] + h[-2]) # numerator of cf
            k.append(qq[i]*k[-1] + k[-2]) # denominator of cf

        for p in k:
            if p == 1 or  p > C:
                continue
            for n in range(1,5): # Check till 5 multiples of cf denominators
                if ((a**(n*p)-1))%C == 0:
                    p = n*p # period
                    sol = 1
                    break
            if sol == 1:
                break
        if sol == 1:
            break
    
    print('C ={}, a = {}, p = {}'.format(C, a, p))
    
    if p%2 !=0:
        print('Try another a')
    elif (a**(p/2)+1)%C == 0:
        print('Try another a')
    else:
        print('Period is:', p)   
        P1 = gcd(int(a**(p/2)+1), C)
        P2 = gcd(int(a**(p/2)-1), C)
        print('Factors are:', P1, P2)
