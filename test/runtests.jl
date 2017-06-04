haskey(ENV, "ONLY_LOAD") && exit()

using CUDAdrv
using Base.Test

using Compat

include("util.jl")

@testset "CUDAdrv" begin

include("pointer.jl")
include("base.jl")

@test length(devices()) > 0
if length(devices()) > 0
    # pick most recent device (based on compute capability)
    global dev = nothing
    for newdev in devices()
        if dev == nothing || capability(newdev) > capability(dev)
            dev = newdev
        end
    end
    info("Testing using device $(name(dev))")
    global ctx = CuContext(dev, CUDAdrv.SCHED_BLOCKING_SYNC)

    @testset "API wrappers" begin
        include("errors.jl")
        include("version.jl")
        include("devices.jl")
        include("context.jl")
        include("module.jl")
        include("memory.jl")
        include("stream.jl")
        include("execution.jl")
        include("events.jl")
        include("profile.jl")
    end

    include("array.jl")
    include("gc.jl")

    include("examples.jl")
    include("documentation.jl")
end

end
