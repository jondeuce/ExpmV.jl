using BenchmarkTools, SparseArrays, LinearAlgebra
using ExpmV

SUITE = BenchmarkGroup()

dimensions = 2 .^ (5:10)
density = 10. .^ (-4:-1)

for p in density
    SUITE[p] = BenchmarkGroup(["density"])
    for d in dimensions
        SUITE[p][d] = @benchmarkable ExpmV.expmv(rt,r,rv) setup=((rt,r,rv)=(rand(), sprandn(ComplexF64,$d,$d,$p), normalize(randn(ComplexF64,$d))))
    end
end
