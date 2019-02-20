using BenchmarkTools, SparseArrays, LinearAlgebra, LinearMaps
import ExpmV, Expokit

#BenchmarkTools.DEFAULT_PARAMETERS.seconds = 120

const p = 1e-3
const m = 8
const dt = 0.5 + 0.1*rand() # time 

SUITE = BenchmarkGroup()

# SUITE["Expm"] = BenchmarkGroup()
SUITE["ExpmV"] = BenchmarkGroup()
SUITE["Expokit"] = BenchmarkGroup()
SUITE["ExpmV_LinMap"] = BenchmarkGroup()
SUITE["Expokit_LinMap"] = BenchmarkGroup()
SUITE["ExpmV_Precomp"] = BenchmarkGroup()
SUITE["Expokit_Precomp"] = BenchmarkGroup()
SUITE["Expokit_LinMap_Precomp"] = BenchmarkGroup()

dimensions = [d for d in 2 .^ (5:10)]

for d in dimensions
    #SUITE[d] = BenchmarkGroup()
    A, b = sprandn(ComplexF64,d,d,m/d), normalize(randn(ComplexF64,d))
    L = LinearMap(A)

    # SUITE["Expm"][d]            = @benchmarkable exp(dt_A)*$b setup=(dt_A = Matrix($dt*$A))
    SUITE["ExpmV"][d]           = @benchmarkable ExpmV.expmv($dt,$A,$b;shift=true)
    SUITE["Expokit"][d]         = @benchmarkable Expokit.expmv($dt,$A,$b)
    SUITE["ExpmV_LinMap"][d]    = @benchmarkable ExpmV.expmv($dt,$L,$b;shift=false)
    SUITE["Expokit_LinMap"][d]  = @benchmarkable Expokit.expmv($dt,$L,$b)

    f, b1, b2 = similar(b), similar(b), similar(b)
    M, _ = ExpmV.select_taylor_degree(A,1)
    SUITE["ExpmV_Precomp"][d]   = @benchmarkable ExpmV.expmv!($f,$dt,$A,$b,$M,$norm,$opnorm,$b1,$b2)
    SUITE["Expokit_Precomp"][d] = @benchmarkable Expokit.expmv!($f,$dt,$A,$b)
    SUITE["Expokit_LinMap_Precomp"][d] = @benchmarkable Expokit.expmv!($f,$dt,$A,$b)
end

results = run(SUITE, verbose = true, seconds = 5)

for d in dimensions
    results_d = results[@tagged d]
    names = collect(keys(results_d))
    len = maximum(length, names)
    sort!(names; by = k -> minimum(results[k][d]).time)
    
    @show d
    for name in names
        print(rpad(name * ": ", len+2))
        show(results[name][d])
        println("")
    end
    println("")
end
