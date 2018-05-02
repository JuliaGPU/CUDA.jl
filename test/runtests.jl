using CUDAnative, CUDAdrv
import LLVM

using Test
import Pkg

@testset "CUDAnative" begin

include("util.jl")

include("base.jl")
include("pointer.jl")
include("codegen.jl")

if CUDAnative.configured
    @test length(devices()) > 0
    if length(devices()) > 0
        # the API shouldn't have been initialized
        @test CuCurrentContext() == nothing

        # now cause initialization
        Mem.alloc(1)
        @test CuCurrentContext() != nothing
        @test device(CuCurrentContext()) == CuDevice(0)

        device!(CuDevice(0))
        device!(CuDevice(0)) do
            nothing
        end

        # test the device selection functionality
        if length(devices()) > 1
            device!(1) do
                @test device(CuCurrentContext()) == CuDevice(1)
            end
            @test device(CuCurrentContext()) == CuDevice(0)

            device!(1)
            @test device(CuCurrentContext()) == CuDevice(1)
        end

        # pick most recent device (based on compute capability)
        global dev = nothing
        for newdev in devices()
            if dev == nothing || capability(newdev) > capability(dev)
                dev = newdev
            end
        end
        @info("Testing using device $(name(dev))")
        global ctx = CuContext(dev, CUDAdrv.SCHED_BLOCKING_SYNC)

        if capability(dev) < v"2.0"
            @warn("native execution not supported on SM < 2.0")
        else
            include("device/codegen.jl")
            include("device/execution.jl")
            include("device/pointer.jl")
            include("device/array.jl")
            include("device/intrinsics.jl")

            include("examples.jl")
            if "Documenter" in keys(Pkg.installed())
                include("documentation.jl")
            else
                @warn("Documenter.jl not installed, skipping CUDAnative documentation tests.")
            end
        end
    end
else
    @warn("CUDAnative.jl has not been configured; skipping on-device tests.")
end

end
