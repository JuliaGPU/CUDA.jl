module Deps

Base.Experimental.@compiler_options compile=min optimize=0

import ..CUDA
import ..LLVM

include("discovery.jl")
include("compatibility.jl")
include("bindeps.jl")
include("utils.jl")

end
