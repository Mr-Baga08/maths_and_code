"""
In-Place Modification (Zero Extra Memory). 
In academia, you write down separate $L$ and $U$ matrices.
In production code, allocating memory for two new n * n matrices is incredibly wasteful. 
Because L is strictly lower triangular (we know its diagonal is all 1s) 
and $U$ is upper triangular, they perfectly slot together into a single matrix. 
The algorithm simply overwrites the original matrix $A$ as it goes.
"""

"""
Vectorized Submatrix Updates
Instead of using slow, nested for loops to update elements one by one, 
optimized libraries update the entire remaining block of the matrix 
simultaneously using vector hardware instructions (SIMD).
"""

"""
Bounding the Multipliers (Partial Pivoting)
How partial pivoting prevents roundoff errors. 
A computer has finite floating-point precision. 
If your pivot a_kk is a very tiny number (like 10^{-10}), 
and you divide by it to find your multiplier, that multiplier becomes massive. 
When you later subtract that massive number from a normal-sized number in the matrix, 
the smaller numbers are effectively truncated and 
lost forever—a phenomenon called catastrophic cancellation.

By searching the current column for the largest absolute value and 
swapping it to the pivot position, we guarantee that all 
our multipliers are strictly less than 1. 
This keeps the floating-point arithmetic highly stable.
"""


import numpy as np

def lu_factorization_optimised(A):
    A = A.astype(float) 
    n = A.shape[0]

    p = np.arange(n) 

    for k in range(n-1):
        pivot_index = np.argmax(np.abs(A[k:, k])) + k
        
        if pivot_index != k:
            A[[k, pivot_index], :] = A[[pivot_index, k], :] # A crucial rule of advanced indexing is that it always creates a copy of the data in the background.
            p[[k, pivot_index]] = p[[pivot_index, k]]

        if np.abs(A[k, k]) < 1e-12:
            raise ValueError("ill-conditioned matrix detected during pivoting")
            
        A[k+1:, k] = A[k+1:, k] / A[k, k]
        A[k+1:, k+1:] -= np.outer(A[k+1:, k], A[k, k+1:])

    return A, p

# Example usage     
if __name__ == "__main__":
    A = np.array([[4, 3], [6, 3]], dtype=float)
    LU, p = lu_factorization_optimised(A)
    print("LU Matrix:\n", LU)
    print("Pivot Indices:", p)