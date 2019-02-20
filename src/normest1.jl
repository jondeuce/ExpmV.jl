function ht21(A,debug=false;itmax=20)
    n,m = size(A)
    T = eltype(A)
    Tr = real(T)
    if n != m
        error("norm 1 estimation can only be applied to square matrices")
    end
    x = ones(T,n)/n
    y, z, ξ = similar(x), similar(x), similar(x)
    γ = 0
    itcount = itmax
    for k in 1:itmax
        y = mul!(y, A, x) #y = A*x
        ξ .= sign.(y) #ξ = sign(y)
        z = mul!(z, A', ξ) #z = A'*ξ
        z_dot_x = dot(z, x)
        if debug
            # z_dot_x should be real to within floating point errors
            @show z_dot_x
            @assert abs(imag(z_dot_x)) < 10*eps(real(z_dot_x))
        end
        if norm(z,Inf) <= real(z_dot_x) && k > 1
            γ = norm(y, 1)
            itcount = k
            break
        end
        x = fill!(x, 0) #x = zeros(Float64,n);
        x[lazyfindmax(abs, z)[2]] = 1 #x[indmax(abs(z))] = 1.
    end
    
    # b = Float64[(-1)^(i+1) * (1+(i-1)/(n-1)) for i in 1:n]
    # out = max(γ,norm(A*b,1)/norm(b,1))
    for i in 1:n
        sgn = ifelse(iseven(i+1), 1, -1)
        x[i] = sgn * (1 + Tr(i-1) / Tr(n-1))
    end
    y = mul!(y, A, x)
    est = max(γ, norm(y,1) / norm(x,1))

    return est, itcount
end

function ht22(A,t::Int=2,debug=false;itmax=10)
    n,m = size(A)
    T = eltype(A)
    Tr = real(T)
    if n!=m
        error("norm 1 estimation can only be applied to square matrices")
    end
    # start with a random matrix of normalized columns
    X = randn(T,n,t)
    Y, Z = similar(X), similar(X)
    for col=1:t
        # scale!(slice(X,1:n,col),1/norm(slice(X,1:n,col),1))
        @views X[:,col] .*= 1/norm(X[:,col],1)
    end
    # Id = Array(I,n,n) #Id = eye(n)
    
    h = zeros(Tr,n)
    itcount = itmax
    for k in 1:itmax
        Y = mul!(Y,A,X) #Y = A*X
        @views g = Tr[norm(Y[:,j],1) for j in 1:t]
        _, ind_best = findmax(g)
        g = sort!(g; rev=true)
        if debug
            @show g[1]
        end
        S = sign.(Y)
        Z = mul!(Z,A',S) #Z = A'*S
        @views h = Tr[maximum(abs, Z[i,:]) for i in 1:n] #Float64[maximum(abs(vec(Z[i,:]))) for i in 1:n]
        ind = [1:n;]
        h_ind = sortslices(hcat(h,typeof(h)(ind)); dims=1, by=first, rev=true) #sortrows(hcat(h,ind),by=x->x[1],rev=true)
        h = copyto!(h, h_ind[:,1])
        ind = round.(Int, h_ind[:,2])
        
        @views Z_dot_X = dot(Z[:,ind_best], X[:,ind_best])
        if debug
            # Z_dot_X should be real to within floating point errors
            @show Z_dot_X
            @assert abs(imag(Z_dot_X)) < 10*eps(real(Z_dot_X))
        end
        if maximum(h) <= real(Z_dot_X)
            itcount = k
            break
        end
        if debug
            @show h[1]
        end
        for j in 1:t
            @views fill!(X[:,j], 0) # X[:,j] = Id[:,ind[j]]
            X[ind[j],j] = 1
        end
    end
    est = h[1]
    
    return est, itcount
end

function ht23(A,t::Int=2;itmax=2)
    n,m = size(A)
    T = eltype(A)
    Tr = real(T)
    if n!=m
        error("norm 1 estimation can only be applied to square matrices")
    end
    # start with a random matrix of normalized columns
    X = randn(T,n,t)
    Y, Z = similar(X), similar(X)
    for col=1:t
        @views X[:,col] .*= 1/norm(X[:,col],1)
    end
    # Id = eye(n)
    est_old = zero(Tr) #0 # initial old estimate
    est = zero(Tr) #0 # initial estimate
    # ind = zeros(Int,n)
    S = zeros(T,n,t) #S = zeros(Int8,n,t)
    S_old = zeros(T,n,t) #S_old = zeros(n,t)
    ind_hist = [] # integer vector recording indices of used unit vectors
    itcount = itmax
    for k=1:itmax+1
        Y = mul!(Y,A,X) #Y = A*X # TODO: make Y preallocated, and in place multiplication
        # est, est_indx = findmax([norm(slice(Y,1:n,col),1) for col in 1:t])
        est, est_indx = lazyfindmax(y->norm(y,1), view(Y,:,col) for col in 1:t)
        if est > est_old || k == 2
            ind_best = est_indx
            # w = Y[:,ind_best]
        end
        if est <= est_old && k >= 2
            est = est_old
            itcount = k
            break
        end
        est_old = est
        S_old = copyto!(S_old, S)
        if k > itmax
            break
        end
        S .= sign.(Y) #copy!(S,sign(Y))
        # If every column of S is parallel to a column of S_old, break
        if parallel_cols(S, S_old)
            itcount = k
            break
        end
        if t>1
            randomize_parallel!(S,S_old)
        end
        Z = mul!(Z,A',S) #Z = transpose(A)*S
        @views h = Tr[norm(Z[i,:], Inf) for i in 1:n]
        if k>=2 && maximum(h) == h[ind_best]
            itcount = k
            break
        end
        # Sort h in decreasing order, and reorder ind correspondingly
        ind = [1:n;]
        h_ind = sortslices(hcat(h,typeof(h)(ind)); dims=1, by=first, rev=true) #sortrows(hcat(h,ind),by=x->x[1],rev=true)
        h = h_ind[:,1]
        #ind = integer(h_ind[:,2]) 
        ind = round.(Int, h_ind[:,2]) #ind = @compat round(Integer,h_ind[:,2])
        if t>1
            if ind[1:t] ⊆ ind_hist
                itcount = k
                break
            end
            # replace ind[1:t] with the first t indices in
            # ind[1:n] that are not in ind_hist
            ind[1:t] = filter(x->!(x ⊆ ind_hist),ind)[1:t]
        end
        for j in 1:t
            @views fill!(X[:,j], 0) # X[:,j] = Id[:,ind[j]]
            X[ind[j],j] = 1
        end
        ind_hist = [ind_hist; ind[1:t]]
    end

    return est, itcount
end

function parallel_cols(S,So)
    n = size(S,1)
    # the assumption is that S,So are matrices with elements ±1,
    # so being parallel would translate to abs value of inner product ≈ n
    return any(map(x->isapprox(x,n), abs.(So'*S)); dims = 1) |> all
end

function randomize_parallel!(S,So)
    n = size(S,1)
    # Ensure no column of S is parallel to a column of S
    ips = map(x->isapprox(x,n), abs.(S'*S))
    for i = 1:size(ips,1) #1:n
        ips[i,i] = false
        if any(ips[:,i])
            @views S[:,i] .= rand([-1,1], n)
        end
    end
    # Ensure no column of S is parallel to a column of So
    ips = map(x->isapprox(x,n), abs.(So'*S))
    for i = 1:size(ips,1) #1:n
        ips[i,i] = false
        if any(ips[:,i])
            @views S[:,i] .= rand([-1,1], n)
        end
    end
end

# using BenchmarkTools
# using LinearMaps
# 
# function print_ests(A,t,debug=false)
#     # println("ht21: $(ht21(A,debug)[1])")
#     # println("ht22: $(ht22(A,t,debug)[1])")
#     # println("ht23: $(ht23(A,t)[1])")
#     # println("norm: $(opnorm(Array(A),1))")
#     println("ht21: $(@btime ht21($A,$debug)[1] evals = 3)")
#     println("ht22: $(@btime ht22($A,$t,$debug)[1] evals = 3)")
#     println("ht23: $(@btime ht23($A,$t)[1] evals = 3)")
#     println("norm: $(@btime opnorm(Array($A),1) evals = 3)")
#     return nothing
# end
# 
# function test_normest1(t=2; d=100, p=6/d)
#     println("Full...")
#     A = randn(Float64,d,d)
#     ests = print_ests(A,t)
# 
#     println("Complex Full...")
#     A = randn(ComplexF64,d,d)
#     ests = print_ests(A,t)
# 
#     println("Sparse...")
#     A = sprandn(Float64,d,d,p)
#     ests = print_ests(A,t)
# 
#     println("Complex Sparse...")
#     A = sprandn(ComplexF64,d,d,p)
#     ests = print_ests(A,t)
# 
#     println("Complex LinearMaps...")
#     A = randn(ComplexF64,d,d) |> LinearMap
#     ests = print_ests(A,t)
# 
#     return nothing
# end