# norm1est.jl is adapted from
# https://github.com/JuliaLang/julia/blob/master/base/sparse/linalg.jl
# and is licensed under the MIT License.
#
# Copyright (c) 2009-2015: Jeff Bezanson, Stefan Karpinski, Viral
# B. Shah, and other contributors:
#
# https://github.com/JuliaLang/julia/contributors
#
#   Permission is hereby granted, free of charge, to any person
#   obtaining a copy of this software and associated documentation
#   files (the "Software"), to deal in the Software without
#   restriction, including without limitation the rights to use, copy,
#   modify, merge, publish, distribute, sublicense, and/or sell copies
#   of the Software, and to permit persons to whom the Software is
#   furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
# BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
# ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

function A_pow_n_B!(res, x, n::Int, y)
    if n < 1
        error("")
    end
    tmp = similar(y)
    #tmp2 = similar(y)
    mul!(res, x, y)
    for i in 1:n-1
        mul!(tmp, x, res)
        copyto!(res, tmp)
    end
    #copyto!(res, tmp)
end

function At_pow_n_B!(res,x,n::Int,y)
    if n < 1
        error("")
    end
    tmp = similar(y)
    mul!(res, transpose(x), y)
    for i in 1:n-1
        mul!(tmp, transpose(x), res)
        copyto!(res, tmp)
    end
end

function norm1est(m::Integer, A::SparseMatrixCSC{T}, t::Integer = min(2,maximum(size(A)))) where T
    maxiter = 5
    # Check the input
    if size(A,1) != size(A,2)
        error("Matrix exponential only defined for square matrices")
    end
    n = size(A,1)
    if t <= 0
        throw(ArgumentError("number of blocks must be a positive integer"))
    end
    if t > n
        throw(ArgumentError("number of blocks must not be greater than $n"))
    end
    ind = Array{Int64}(undef, n)
    ind_hist = Array{Int64}(undef, maxiter * t)

    Ti = typeof(float(zero(T)))

    S = zeros(T <: Real ? Int : Ti, n, t)

    function _rand_pm1!(v)
        for i in eachindex(v)
            v[i] = rand() < 0.5 ? 1 : -1
        end
    end

    function _any_abs_eq(v,n::Int)
        for i in eachindex(v)
            if abs(v[i])==n
                return true
            end
        end
        return false
    end

    # Generate the block matrix
    X = Array{Ti}(undef, n, t)
    Y = similar(X)
    Z = similar(X)

    X[1:n,1] .= 1
    for j = 2:t
        while true
            _rand_pm1!(view(X,1:n,j))
            yaux = X[1:n,j]' * X[1:n,1:j-1]
            if !_any_abs_eq(yaux,n)
                break
            end
        end
    end
    rmul!(X, 1. /n)

    iter = 0
    local est
    local est_old
    est_ind = 0

    while iter < maxiter
        iter += 1
        A_pow_n_B!(Y, A, m, X)

        est = zero(real(eltype(Y)))

        est_ind = 0
        for i = 1:t
            y = norm(Y[1:n, i], 1)
            if y > est
                est = y
                est_ind = i
            end
        end
        if iter == 1
            est_old = est
        end
        if est > est_old || iter == 2
            ind_best = est_ind
        end
        if iter >= 2 && est <= est_old
            est = est_old
            break
        end
        est_old = est
        S_old = copy(S)
        for j = 1:t
            for i = 1:n
                S[i,j] = Y[i,j] == 0 ? one(Y[i,j]) : sign(Y[i,j])
            end
        end

        if T <: Real
            # Check wether cols of S are parallel to cols of S or S_old
            for j = 1:t
                while true
                    repeated = false
                    if j > 1
                        saux = S[1:n,j]' * S[1:n,1:j-1]
                        if _any_abs_eq(saux,n)
                            repeated = true
                        end
                    end
                    if !repeated
                        saux2 = S[1:n,j]' * S_old[1:n,1:t]
                        if _any_abs_eq(saux2,n)
                            repeated = true
                        end
                    end
                    if repeated
                        _rand_pm1!(view(S,1:n,j))
                    else
                        break
                    end
                end
            end
        end

        # Use the conjugate transpose
        At_pow_n_B!(Y,A,m,X)
        h_max = zero(real(eltype(Z)))
        h = zeros(real(eltype(Z)), n)
        h_ind = 0
        for i = 1:n
            h[i] = norm(Z[i,1:t], Inf)
            if h[i] > h_max
                h_max = h[i]
                h_ind = i
            end
            ind[i] = i
        end
        if iter >=2 && ind_best == h_ind
            break
        end
        p = sortperm(h, rev=true)
        h = h[p]
        permute!(ind, p)
        if t > 1
            addcounter = t
            elemcounter = 0
            while addcounter > 0 && elemcounter < n
                elemcounter = elemcounter + 1
                current_element = ind[elemcounter]
                found = false
                for i = 1:t * (iter - 1)
                    if current_element == ind_hist[i]
                        found = true
                        break
                    end
                end
                if !found
                    addcounter = addcounter - 1
                    for i = 1:current_element - 1
                        X[i,t-addcounter] = 0
                    end
                    X[current_element,t-addcounter] = 1
                    for i = current_element + 1:n
                        X[i,t-addcounter] = 0
                    end
                    ind_hist[iter * t - addcounter] = current_element
                else
                    if elemcounter == t && addcounter == t
                        break
                    end
                end
            end
        else
            ind_hist[1:t] = ind[1:t]
            for j = 1:t
                for i = 1:ind[j] - 1
                    X[i,j] = 0
                end
                X[ind[j],j] = 1
                for i = ind[j] + 1:n
                    X[i,j] = 0
                end
            end
        end
    end
    return est, iter
end
