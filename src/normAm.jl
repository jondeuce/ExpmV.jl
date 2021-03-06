function normAm(A,
                m,
                norm = LinearAlgebra.norm,
                opnorm = LinearAlgebra.opnorm;
                check_positive = false)::Real
    #NORMAM   Estimate of 1-norm of power of matrix.
    #   NORMAM(A,m) estimates norm(A^m,1).
    #   If A has nonnegative elements the estimate is exact.
    #   [C,MV] = NORMAM(A,m) returns the estimate C and the number MV of
    #   matrix-vector products computed involving A or A^*.

    #   Reference: A. H. Al-Mohy and N. J. Higham, A New Scaling and Squaring
    #   Algorithm for the Matrix Exponential, SIAM J. Matrix Anal. Appl. 31(3):
    #   970-989, 2009.

    #   Awad H. Al-Mohy and Nicholas J. Higham, September 7, 2010.

    t = 2 # Number of columns used by NORMEST1.

    if check_positive # expensive check; forces matrix materialization
        if eltype(A) <: Real #&& hasmethod(opnorm, Tuple{typeof(A), typeof(1)})
            if isequal(A, abs.(A)) #sum(A .< 0) == 0 # for positive matrices only
                n = size(A, 2)
                e = ones(n, 1)
                f = similar(e)
                for j=1:m
                    mul!(f, A, e)
                    copyto!(e, f)
                end
                return norm(e, Inf)
            end
        end
    end
    return norm1est(A, m, t)
end
