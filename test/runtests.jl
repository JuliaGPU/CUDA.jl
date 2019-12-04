using CUDAdrv

using Test

@testset "CUDAdrv" begin

include("util.jl")
include("array.jl")

include("pointer.jl")

@test CUDAdrv.functional()

@test length(devices()) > 0
if length(devices()) > 0
    @test CuCurrentContext() == nothing

    # pick a suiteable device (by available memory,
    # but also by capability if testing needs to be thorough)
    candidates = [(dev=dev,
                   cap=capability(dev),
                   mem=CuContext(ctx->CUDAdrv.available_memory(), dev))
                  for dev in devices()]
    thorough = parse(Bool, get(ENV, "CI_THOROUGH", "false"))
    if thorough
        sort!(candidates, by=x->(x.cap, x.mem))
    else
        sort!(candidates, by=x->x.mem)
    end
    pick = last(candidates)
    @info("Testing using device $(name(pick.dev)) (compute capability $(pick.cap), $(Base.format_bytes(pick.mem)) available memory) on CUDA driver $(CUDAdrv.version())")
    global dev = pick.dev

    global ctx = CuContext(dev)
    @test CuCurrentContext() != nothing

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
        include("occupancy.jl")
    end

    include("gc.jl")

    include("examples.jl")
end

end
