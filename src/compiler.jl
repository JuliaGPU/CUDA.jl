# JIT compilation of Julia code to PTX

include("compiler/common.jl")
include("compiler/irgen.jl")
include("compiler/optim.jl")
include("compiler/validation.jl")
include("compiler/mcgen.jl")
include("compiler/driver.jl")

function __init_compiler__()
    # enable generation of FMA instructions to mimic behavior of nvcc
    LLVM.clopts("--nvptx-fma-level=1")
end
