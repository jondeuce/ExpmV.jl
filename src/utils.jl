####
#### expmv utils
####
diagshift!(A, mu) = A - mu * I # fallback out-of-place (for sparse) or lazy (for LinearMaps) operator
diagshift!(A::StridedArray, mu) = A .-= mu * I # modify in-place

####
#### norm1est utils
####
function A_pow_n_B!(res, A, n::Integer, v)
    tmp = similar(v)
    mul!(res, A, v)
    for i in 1:n-1
        mul!(tmp, A, res)
        copyto!(res, tmp)
    end
end

function At_pow_n_B!(res, A, n::Integer, v)
    tmp = similar(v)
    mul!(res, adjoint(A), v)
    for i in 1:n-1
        mul!(tmp, adjoint(A), res)
        copyto!(res, tmp)
    end
end

####
#### normest1 utils
####
function lazyfindmax(f,x)
    idx = 1 # default
    maxval = f(first(x)) # default
    for (i,xi) in enumerate(x)
        (i == 1) && continue
        fval = f(xi)
        if fval > maxval
            idx, maxval = i, fval
        end
    end
    return maxval, idx
end
lazyfindmax(x) = lazyfindmax(identity,x)

# extension of Base.sign where x==0 
zerosign(x) = x == zero(x) ? one(x) : Base.sign(x)