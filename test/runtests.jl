using CUDAnative, CUDAdrv
using Base.Test

@testset "CUDAnative" begin

include("util.jl")

include("base.jl")

if CUDAnative.configured
    # requiring a configured LLVM.jl
    include("codegen.jl")

    # requiring a configured CUDAdrv.jl
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
            include("execution.jl")
            include("array.jl")
            include("intrinsics.jl")

            include("examples.jl")
            if "Documenter" in keys(Pkg.installed())
                include("documentation.jl")
            else
                warn("Documenter.jl not installed, skipping documentation tests.")
            end
        end
    end
else
    warn("CUDAnative.jl has not been configured; skipping most tests.")
end

end
