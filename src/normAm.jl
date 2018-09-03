function normAm(A,m)
    #NORMAM   Estimate of 1-norm of power of matrix.
    #   NORMAM(A,m) estimates norm(A^m,1).
    #   If A has nonnegative elements the estimate is exact.
    #   [C,MV] = NORMAM(A,m) returns the estimate C and the number MV of
    #   matrix-vector products computed involving A or A^*.

    #   Reference: A. H. Al-Mohy and N. J. Higham, A New Scaling and Squaring
    #   Algorithm for the Matrix Exponential, SIAM J. Matrix Anal. Appl. 31(3):
    #   970-989, 2009.

    #   Awad H. Al-Mohy and Nicholas J. Higham, September 7, 2010.

    t = 2; # Number of columns used by NORMEST1.

    n = size(A, 1);

    return norm1est(m,A,t)
end
