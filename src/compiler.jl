# JIT compilation of Julia code to PTX

include(joinpath("compiler", "common.jl"))
include(joinpath("compiler", "irgen.jl"))
include(joinpath("compiler", "optim.jl"))
include(joinpath("compiler", "validation.jl"))
include(joinpath("compiler", "rtlib.jl"))
include(joinpath("compiler", "mcgen.jl"))
include(joinpath("compiler", "driver.jl"))

function __init_compiler__()
    # enable generation of FMA instructions to mimic behavior of nvcc
    LLVM.clopts("--nvptx-fma-level=1")
end
