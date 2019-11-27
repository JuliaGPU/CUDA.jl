# JIT compilation of Julia code to PTX

const to = TimerOutput()

timings() = (TimerOutputs.print_timer(to); println())

enable_timings() = (TimerOutputs.enable_debug_timings(CUDAnative); return)

include("compiler/common.jl")
include("compiler/irgen.jl")
include("compiler/optim.jl")
include("compiler/validation.jl")
include("compiler/rtlib.jl")
include("compiler/mcgen.jl")
include("compiler/debug.jl")
include("compiler/driver.jl")

function __init_compiler__()
    # enable generation of FMA instructions to mimic behavior of nvcc
    LLVM.clopts("-nvptx-fma-level=1")

    TimerOutputs.reset_timer!(to)
end
