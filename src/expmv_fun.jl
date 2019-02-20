using LinearAlgebra
using SparseArrays

export expmv

"""
    expmv!(f, t, A, b; <keyword arguments>)

Returns the matrix-vector product ``exp(t A) b`` where `A` is a ``n × n`` sparse
real or complex matrix or matrix-free object, `b` is a vector of ``n`` real or
complex elements, and `t` is a time step parameter (or a `StepRangeLen` object
representing a range of values).

# Arguments
* `f`: an `n`-vector for preallocating the output 
* `t`: `Number` or `StepRangeLen` object
* `A`: a `n × n` real or complex sparse matrix
* `b`: an `n`-vector
* `M = []`: manually set the degree of the Taylor expansion
* `precision = :double`: can be `:double`, `:single` or `:half`.
* `shift = $DEFAULT_SHIFT`: set to `true` to apply a shift in order to reduce the norm of A
        (see Sec. 3.1 of the paper)
* `full_term = false`: set to `true` to evaluate the full Taylor expansion instead
        of truncating when reaching the required precision
* `check_positive = false`: set to `true` to check if `A` has strictly positive entries
        for a faster evaluation of the 1-norm
"""
function expmv!(
        f::AbstractVecOrMat, t::Number, A, b::AbstractVecOrMat,
        M = nothing, norm = LinearAlgebra.norm, opnorm = LinearAlgebra.opnorm,
        b1 = similar(b), b2 = similar(b);
        precision = :double, shift = DEFAULT_SHIFT, full_term = false, check_positive = false
    )
        
    if shift == true && !hasmethod(tr, Tuple{typeof(A)})
        shift = false
        @warn "Shift set to false as $(typeof(A)) doesn't support tr"
    end
    
    n = size(A, 2)
    Tr = real(eltype(b))

    mu = zero(Tr)
    if shift
        mu = tr(A)/n
        A = diagshift!(A, mu) #A = A - mu * I
    end

    if M == nothing
        tt = 1
        (M,alpha,unA) = select_taylor_degree(t*A, size(b,2); precision=precision, shift=shift, check_positive=check_positive)
    else
        tt = t
    end

    tol =
      if precision == :double
          Tr(2.0^(-53))
      elseif precision == :single
          Tr(2.0^(-24))
      elseif precision == :half
          Tr(2.0^(-10))
      end

    s = 1

    if t == 0
        m = 0
    else
        (m_max,p) = size(M)
        U = diagm(0 => 1:m_max)
        C = ((ceil.(abs.(tt)*M))'*U )

        C[C .== 0] .= Inf

        cost, idx = findmin(C')
        # idx is a CarthesianIndex if C' is a Matrix, or a scalar if C' is a row
        # vector. idx[1] extract the first index, i.e. row, of the CarthesianIndex
        m = idx[1]

        if cost == Inf
            cost = 0
        end

        s = max(cost/m,1)
    end

    eta = 1

    if shift
        eta = exp(t*mu/s)
    end

    f = copyto!(f, b)
    b1 = copyto!(b1, b)
    b2 = copyto!(b2, b)
    c1 = c2 = Tr(Inf)

    @inbounds for i = 1:s
        if !full_term
            c1 = norm(b1, Inf) # only need to update if !full_term
        end

        for k = 1:m
            b2 = mul!(b2, A, b1)
            b2 .*= (t / (s * k)) # b = (t/(s*k))*(A*b)
            b1, b2 = b2, b1
            f .+= b1 # f = axpy!(1, b1, f)

            if !full_term
                c2 = norm(b1, Inf) # only need to update if !full_term
                if c1 + c2 <= tol * norm(f, Inf)
                    break
                end
                c1 = c2
            end

        end

        if shift
            f .*= eta
        end

        b1 = copyto!(b1, f)
    end

    return f
end

function expmv(
        t::Number, A, b::AbstractVecOrMat,
        M = nothing, norm = LinearAlgebra.norm, opnorm = LinearAlgebra.opnorm,
        b1 = copy(b), b2 = similar(b);
        precision = :double, shift = DEFAULT_SHIFT, full_term = false, check_positive = false
    )

    return expmv!(
        similar(b), t, A, b, M, norm, opnorm, b1, b2;
        precision = precision, shift = shift, full_term = full_term, check_positive = check_positive
    )
end