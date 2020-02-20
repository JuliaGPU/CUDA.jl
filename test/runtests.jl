using Test

using CUDAnative, CuArrays, CUDAdrv
import Adapt, LLVM

include("util.jl")

@testset "CUDAnative" begin

@test CUDAnative.functional()

CUDAnative.version()
CUDAnative.release()

CUDAnative.enable_timings()

# see if we have a device to run tests on
# (do this early so that the codegen tests can target the same compute capability)
@test length(devices()) > 0
if length(devices()) > 0
    # the API shouldn't have been initialized
    @test CuCurrentContext() == nothing

    callback_data = nothing
    CUDAnative.atcontextswitch() do tid, ctx
        callback_data = (tid, ctx)
    end

    # now cause initialization
    ctx = context()
    @test CuCurrentContext() === ctx
    @test device() == CuDevice(0)
    @test callback_data[1] == Threads.threadid()
    @test callback_data[2] === ctx

    device!(CuDevice(0))
    device!(CuDevice(0)) do
        nothing
    end

    device_reset!()

    # test the device selection functionality
    if length(devices()) > 1
        device!(0)
        device!(1) do
            @test device() == CuDevice(1)
        end
        @test device() == CuDevice(0)

        device!(1)
        @test device() == CuDevice(1)
    end

    # pick a suiteable device
    candidates = [(dev=dev,
                   cap=capability(dev),
                   mem=CuContext(ctx->CUDAdrv.available_memory(), dev))
                  for dev in devices()]
    ## pick a device that is fully supported by our CUDA installation, or tools can fail
    ## NOTE: we don't reuse target_support[] which is also bounded by LLVM support,
    #        and is used to pick a codegen target regardless of the actual device.
    cuda_support = CUDAnative.cuda_compat()
    filter!(x->x.cap in cuda_support.cap, candidates)
    ## order by available memory, but also by capability if testing needs to be thorough
    thorough = parse(Bool, get(ENV, "CI_THOROUGH", "false"))
    if thorough
        sort!(candidates, by=x->(x.cap, x.mem))
    else
        sort!(candidates, by=x->x.mem)
    end
    isempty(candidates) && error("Could not find any suitable device for this configuration")
    pick = last(candidates)
    @info("Testing using device $(name(pick.dev)) (compute capability $(pick.cap), $(Base.format_bytes(pick.mem)) available memory) on CUDA driver $(CUDAdrv.version()) and toolkit $(CUDAnative.version())")
    device!(pick.dev)
end

include("base.jl")
include("pointer.jl")
include("codegen.jl")

if isempty(devices())
    @warn("No CUDA-capable devices available; skipping on-device tests.")
else
    if capability(device()) < v"2.0"
        @warn("native execution not supported on SM < 2.0")
    else
        include("device/codegen.jl")
        include("device/execution.jl")
        include("device/pointer.jl")
        include("device/array.jl")
        include("device/cuda.jl")
        include("device/wmma.jl")

        include("nvtx.jl")

        include("examples.jl")
    end
end

haskey(ENV, "CI") && CUDAnative.timings()

end
