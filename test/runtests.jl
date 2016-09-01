using CUDAnative, CUDAdrv
using Base.Test

@test devcount() > 0

include("compilation.jl")
include("codegen.jl")

macro on_device(exprs)
    quote
        @target ptx function kernel()
            $exprs

            return nothing
        end

        @cuda (1,1) kernel()
        synchronize(default_stream())
    end
end

dev = CuDevice(0)
if capability(dev) < v"2.0"
    warn("native execution not supported on SM < 2.0")
else
    ctx = CuContext(dev)

    include("execution.jl")
    include("array.jl")
    include("intrinsics.jl")

    destroy(ctx)
end
