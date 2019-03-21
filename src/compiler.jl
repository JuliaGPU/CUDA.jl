# JIT compilation of Julia code to PTX

const to = Ref{TimerOutput}()

function timings!(new=TimerOutput())
  global to
  to[] = new
  return
end

timings() = (TimerOutputs.print_timer(to[]; allocations=false); println())

include(joinpath("compiler", "common.jl"))
include(joinpath("compiler", "irgen.jl"))
include(joinpath("compiler", "optim.jl"))
include(joinpath("compiler", "validation.jl"))
include(joinpath("compiler", "rtlib.jl"))
include(joinpath("compiler", "mcgen.jl"))
include(joinpath("compiler", "debug.jl"))
include(joinpath("compiler", "driver.jl"))

function __init_compiler__()
    # enable generation of FMA instructions to mimic behavior of nvcc
    LLVM.clopts("--nvptx-fma-level=1")

    timings!()
end
