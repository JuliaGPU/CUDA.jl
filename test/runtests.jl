using CUDAnative, CUDAdrv
import LLVM

using Test
import Pkg

@testset "CUDAnative" begin

include("util.jl")

include("base.jl")
include("pointer.jl")

if LLVM.configured
    include("codegen.jl")
else
    warn("LLVM.jl has not been configured; skipping CUDAnative codegen tests.")
end

if CUDAnative.configured
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

        if capability(dev) < v"2.0"
            warn("native execution not supported on SM < 2.0")
        else
            include("codegen_device.jl")
            include("execution.jl")
            include("array.jl")
            include("intrinsics.jl")

            include("examples.jl")
            if "Documenter" in keys(Pkg.installed())
                include("documentation.jl")
            else
                warn("Documenter.jl not installed, skipping CUDAnative documentation tests.")
            end
        end
    end
else
    warn("CUDAnative.jl has not been configured; skipping CUDAnative execution tests.")
end

end
