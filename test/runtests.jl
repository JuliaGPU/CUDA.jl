using Test

using CUDAnative, CuArrays, CUDAdrv
import Adapt, LLVM

include("util.jl")

@testset "CUDAnative" begin

@test CUDAnative.functional()

CUDAnative.version()

CUDAnative.enable_timings()

# see if we have a device to run tests on
# (do this early so that the codegen tests can target the same compute capability)
global dev, cap
dev = nothing
@test length(devices()) > 0
if length(devices()) > 0
    # the API shouldn't have been initialized
    @test CuCurrentContext() == nothing

    callback_data = nothing
    CUDAnative.atcontextswitch() do tid, ctx, dev
        callback_data = (tid, ctx, dev)
    end

    # now cause initialization
    ctx = CuGetContext()
    @test CuCurrentContext() === ctx
    @test device(ctx) == CuDevice(0)
    @test callback_data[1] == Threads.threadid()
    @test callback_data[2] === ctx
    @test callback_data[3] == CuDevice(0)

    device!(CuDevice(0))
    device!(CuDevice(0)) do
        nothing
    end

    device_reset!()

    # test the device selection functionality
    if length(devices()) > 1
        device!(1) do
            @test device(CuCurrentContext()) == CuDevice(1)
        end
        @test device(CuCurrentContext()) == CuDevice(0)

        device!(1)
        @test device(CuCurrentContext()) == CuDevice(1)
    end

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
    @info("Testing using device $(name(pick.dev)) (compute capability $(pick.cap), $(Base.format_bytes(pick.mem)) available memory) on CUDA driver $(CUDAdrv.version()) and toolkit $(CUDAnative.version())")
    device!(pick.dev)
    dev = pick.dev
end
cap = CUDAnative.current_capability()

include("base.jl")
include("pointer.jl")
include("codegen.jl")

if dev === nothing
    @warn("No CUDA-capable devices available; skipping on-device tests.")
else
    if capability(dev) < v"2.0"
        @warn("native execution not supported on SM < 2.0")
    else
        include("device/codegen.jl")
        include("device/execution.jl")
        include("device/pointer.jl")
        include("device/array.jl")
        include("device/cuda.jl")

        include("examples.jl")
    end
end

CUDAnative.timings()

end
