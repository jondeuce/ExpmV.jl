using BenchmarkTools, SparseArrays, LinearAlgebra
import ExpmV, Expokit

#BenchmarkTools.DEFAULT_PARAMETERS.seconds = 120

const p = 1e-3
const dt = 0.5 + 0.1*rand() # time 

SUITE = BenchmarkGroup()

SUITE["Expm"] = BenchmarkGroup()
SUITE["ExpmV"] = BenchmarkGroup()
SUITE["Expokit"] = BenchmarkGroup()

dimensions = [d for d in 2 .^ (5:10)]

for d in dimensions
    #SUITE[d] = BenchmarkGroup()
    SUITE["Expm"][d] = @benchmarkable exp(full_r)*rv setup=((full_r,rv) = (Matrix(dt*sprandn(ComplexF64,$d,$d,p)), normalize(randn(ComplexF64,$d))))
    SUITE["ExpmV"][d] = @benchmarkable ExpmV.expmv(rt,r,rv) setup=((rt,r,rv)=(dt, sprandn(ComplexF64,$d,$d,p), normalize(randn(ComplexF64,$d))))
    SUITE["Expokit"][d] = @benchmarkable Expokit.expmv(rt,r,rv) setup=((rt,r,rv)=(dt, sprandn(ComplexF64,$d,$d,p), normalize(randn(ComplexF64,$d))))
end

results = run(SUITE, verbose = true, seconds = 10)
