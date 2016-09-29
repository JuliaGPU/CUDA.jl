using CUDAnative, CUDAdrv
using Base.Test

@test devcount() > 0

include("codegen.jl")

macro on_device(dev, exprs)
    quote
        @target ptx function kernel()
            $exprs

            return nothing
        end

        @cuda $dev (1,1) kernel()
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
