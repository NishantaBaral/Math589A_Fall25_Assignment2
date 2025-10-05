
import numpy as np
import paqlu_decomposition_rectangular as paqlu_rectangular
import paqlu_decomposition_square as paqlu_square


def plu_decomposition_in_place(A):
    m,n = A.shape
    if m == n:
        P,A,L,U = paqlu_square.paqlu_decomposition_in_place(A)
        p = P.argmax(axis=1)   # row-wise argmax -> row permutation
        q = Q.argmax(axis=0)
        return p,q
    else:
        P,Q,L,U = paqlu_rectangular.paqlu_decomposition_in_place(A)
        p = P.argmax(axis=1)   # row-wise argmax -> row permutation
        q = Q.argmax(axis=0)   # col-wise argmax -> col permutation
        return p,q
    
