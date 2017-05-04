using CUDAdrv
using Base.Test

using Compat

include("util.jl")

@testset "CUDAdrv" begin

include("pointer.jl")
include("errors.jl")
include("base.jl")

@test devcount() > 0
if devcount() > 0
    # pick most recent device (based on compute capability)
    global dev = nothing
    for i in 0:devcount()-1
        newdev = CuDevice(i)
        if dev == nothing || capability(newdev) > capability(dev)
            dev = newdev
        end
    end
    info("Testing using device $(name(dev))")
    global ctx = CuContext(dev, CUDAdrv.SCHED_BLOCKING_SYNC)

    include("wrappers.jl")
    include("gc.jl")

    include("examples.jl")
end
end