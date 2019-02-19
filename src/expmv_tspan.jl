using LinearAlgebra
using SparseArrays

function expmv!(
        f::AbstractVecOrMat, t::StepRangeLen, A, b::AbstractVecOrMat,
        M = nothing, norm = LinearAlgebra.norm, opnorm = LinearAlgebra.opnorm;
        precision = :double, shift = false, check_positive = false
    )

    n = size(A, 2)
    T = eltype(b)
    Tr = real(T)
    q = t.len - 1
    
    t0 = Tr(t.ref)
    tmax = Tr(t.ref + (t.len - 1.) * t.step)
    
    if shift == true && !hasmethod(tr, typeof(A))
        shift = false
        @warn "Shift set to false as $(typeof(A)) doesn't support tr"
    end

    force_estm = !hasmethod(opnorm, Tuple{typeof(A), typeof(1)})
    Anorm_dt = (tmax - t0) * normAm(A, 1; check_positive = check_positive);

    if (precision == :single || precision == :half && Anorm_dt > 85.496) || ( precision == :double && Anorm_dt > 63.152)
       force_estm = true;
    end

    if M == nothing
        (M, alpha, unA) = select_taylor_degree(A, size(b,2); precision=precision, shift=shift, force_estm=force_estm, check_positive=check_positive)
    end

    tol =
      if precision == :double
          Tr(2.0)^(-53)
      elseif precision == :single
          Tr(2.0)^(-24)
      elseif precision == :half
          Tr(2.0)^(-10)
      end

    X = zeros(T, n, q+1);
    (m_max, p) = size(M);
    U = diagm(0 => 1:m_max);

    _, s = degree_selector(tmax - t0, M, U, p)
    h = (tmax - t0)/q;

    expmv!(view(X,:,1), t0, A, b, M, norm, opnorm;
        precision=precision, shift=shift, check_positive=check_positive);

    mu = zero(Tr)
    if shift
        mu = tr(A)/n
        A = A - mu * I
    end

    d = max(1, Integer(floor(q/s)))
    j = Integer(floor(q/d))
    r = q - d * j
    m_opt, _ = degree_selector(d, M, U, p)
    dr = d
    
    z = X[:,1]
    K = zeros(T, n, m_opt+1)
    temp = similar(b)

    for i = 1:j+1
        if i > j
            dr = r
        end
        @views K[:,1] .= z
        m = 0
        for k = 1:dr
            f = copyto!(f, z)
            c1 = norm(z, Inf)

            p = 1
            for p = 1:m_opt # outer?
                if p > m
                    temp = mul!(temp, A, view(K,:,p)) # A*K[:,p]
                    @views K[:,p+1] .= (h/p) .* temp
                end

                @views temp .= (Tr(k)^p) .* K[:,p+1]
                f .+= temp
                c2 = norm(temp, Inf)
                if c1 + c2 <= tol * norm(f, Inf)
                    break
                end
                c1 = c2
            end

            m = max(m,p)
            @views X[:, k + (i-1)*d + 1] .= exp(k*h*mu) .* f
        end

        if i <= j
            @views z .= X[:, i*d + 1]
        end
    end
    
    return X
end

function expmv(
        t::StepRangeLen, A, b::AbstractVecOrMat,
        M = nothing, norm = LinearAlgebra.norm, opnorm = LinearAlgebra.opnorm;
        precision = :double, shift = false, check_positive = false
    )

    return expmv!(
        similar(b), t, A, b, M, norm, opnorm;
        precision = precision, shift = shift, check_positive = check_positive
    )
end