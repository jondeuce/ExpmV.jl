module ExpmV

# defaults
const DEFAULT_SHIFT = true

# package code goes here
include("utils.jl")
include("expmv_fun.jl")
include("degree_selector.jl")
include("normAm.jl")
include("expmv_tspan.jl")
include("select_taylor_degree.jl")
include("norm1est.jl")
include("normest1.jl")

end # module
